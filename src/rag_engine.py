# -*- coding: utf-8 -*-
"""
rag_engine.py
-------------
Adaptive RAG Industrial Root Cause Analysis (RCA) Engine
Powered by:
- ChromaDB Vector Store (all-MiniLM-L6-v2 embeddings)
- Multi-domain SOP knowledge base (Bearing SOP + Motor SOP)
- LangGraph Adaptive Query Router & Metadata Retrieval
- Optional Cohere Reranker
- Local Ollama LLM (phi3:latest) with real sensor telemetry grounding
"""

import os
import re
import shutil
from typing import TypedDict, List, Dict, Any, Optional

import ollama
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Optional Cohere Reranker
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOP_DIR = os.path.join(BASE_DIR, "sop_docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

LLM_MODEL = "phi3:latest"

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
cohere_client = (
    cohere.Client(COHERE_API_KEY)
    if (COHERE_AVAILABLE and COHERE_API_KEY)
    else None
)

# ============================================================
# DOMAIN MAPPINGS
# ============================================================

BEARING_CLASS_MAPPING = {
    0: "Normal Operating Condition",
    1: "Inner Race Fault (IR007)",
    2: "Ball / Rolling Element Fault (B007)",
    3: "Outer Race Fault (OR007)"
}

BEARING_CODE_MAPPING = {
    0: "NORMAL",
    1: "IR007",
    2: "B007",
    3: "OR007"
}

MOTOR_CLASS_MAPPING = {
    0: "Normal Motor Operation",
    1: "Shaft Misalignment",
    2: "Broken Rotor Bars"
}

MOTOR_CODE_MAPPING = {
    0: "MOTOR_NORMAL",
    1: "MISALIGNMENT",
    2: "BROKEN_ROTOR"
}


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ============================================================
# MULTI-SOP SECTION EXTRACTION
# ============================================================

def extract_bearing_sections(text):
    sections = []
    # 1. INNER RACE
    m1 = re.search(r"(1\.\s*INNER RACE.*?)(?=\n\s*2\.\s*BALL|\Z)", text, re.I | re.S)
    if m1:
        sections.append(Document(
            page_content=m1.group(1).strip(),
            metadata={"domain": "bearing", "fault_code": "IR007", "fault_name": "Inner Race Fault", "class_id": 1}
        ))
    # 2. BALL
    m2 = re.search(r"(2\.\s*BALL.*?)(?=\n\s*3\.\s*OUTER RACE|\Z)", text, re.I | re.S)
    if m2:
        sections.append(Document(
            page_content=m2.group(1).strip(),
            metadata={"domain": "bearing", "fault_code": "B007", "fault_name": "Ball Fault", "class_id": 2}
        ))
    # 3. OUTER RACE
    m3 = re.search(r"(3\.\s*OUTER RACE.*?)(?=\n\s*4\.\s*NORMAL|\Z)", text, re.I | re.S)
    if m3:
        sections.append(Document(
            page_content=m3.group(1).strip(),
            metadata={"domain": "bearing", "fault_code": "OR007", "fault_name": "Outer Race Fault", "class_id": 3}
        ))
    # 4. NORMAL
    m4 = re.search(r"(4\.\s*NORMAL.*?)(?=\n\s*GENERAL INDUSTRIAL|\Z)", text, re.I | re.S)
    if m4:
        sections.append(Document(
            page_content=m4.group(1).strip(),
            metadata={"domain": "bearing", "fault_code": "NORMAL", "fault_name": "Normal Operating Condition", "class_id": 0}
        ))
    # 5. GENERAL
    m5 = re.search(r"(GENERAL INDUSTRIAL MAINTENANCE GUIDELINES.*)", text, re.I | re.S)
    if m5:
        sections.append(Document(
            page_content=m5.group(1).strip(),
            metadata={"domain": "bearing", "fault_code": "GENERAL", "fault_name": "General Maintenance Guidelines", "class_id": -1}
        ))
    return sections


def extract_motor_sections(text):
    sections = []
    # 1. MISALIGNMENT
    m1 = re.search(r"(1\.\s*SHAFT MISALIGNMENT.*?)(?=\n\s*2\.\s*BROKEN ROTOR|\Z)", text, re.I | re.S)
    if m1:
        sections.append(Document(
            page_content=m1.group(1).strip(),
            metadata={"domain": "motor", "fault_code": "MISALIGNMENT", "fault_name": "Shaft Misalignment", "class_id": 1}
        ))
    # 2. BROKEN ROTOR BARS
    m2 = re.search(r"(2\.\s*BROKEN ROTOR BARS.*?)(?=\n\s*3\.\s*NORMAL MOTOR|\Z)", text, re.I | re.S)
    if m2:
        sections.append(Document(
            page_content=m2.group(1).strip(),
            metadata={"domain": "motor", "fault_code": "BROKEN_ROTOR", "fault_name": "Broken Rotor Bars", "class_id": 2}
        ))
    # 3. MOTOR NORMAL
    m3 = re.search(r"(3\.\s*NORMAL MOTOR OPERATION.*)", text, re.I | re.S)
    if m3:
        sections.append(Document(
            page_content=m3.group(1).strip(),
            metadata={"domain": "motor", "fault_code": "MOTOR_NORMAL", "fault_name": "Normal Motor Operation", "class_id": 0}
        ))
    return sections


def load_all_sop_documents():
    documents = []
    if not os.path.exists(SOP_DIR):
        print(f"[WARN] SOP directory not found: {SOP_DIR}")
        return documents

    for filename in sorted(os.listdir(SOP_DIR)):
        filepath = os.path.join(SOP_DIR, filename)
        if not (filename.endswith(".txt") or filename.endswith(".log") or filename.endswith(".pdf")):
            continue
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                full_text = "\n".join([d.page_content for d in docs])
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

            if "motor" in filename.lower():
                extracted = extract_motor_sections(full_text)
            else:
                extracted = extract_bearing_sections(full_text)

            documents.extend(extracted)
            print(f"[INFO] Indexed {len(extracted)} knowledge sections from {filename}")
        except Exception as e:
            print(f"[WARN] Failed to index {filename}: {e}")

    return documents


# ============================================================
# VECTOR STORE BUILDER
# ============================================================

def build_vector_store():
    print("=" * 60)
    print("[INFO] Building Multi-Domain Industrial RAG Vector Database")
    print("=" * 60)

    embeddings = get_embeddings()
    docs = load_all_sop_documents()

    if not docs:
        print("[ERROR] No valid SOP knowledge sections found.")
        return None

    # Remove duplicates
    unique_docs = []
    seen = set()
    for d in docs:
        key = (d.metadata.get("domain", ""), d.metadata.get("fault_code", ""), d.page_content.strip())
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
            print("[INFO] Purged outdated ChromaDB index.")
        except Exception as e:
            print(f"[WARN] Error resetting ChromaDB: {e}")

    vector_db = Chroma.from_documents(
        documents=unique_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"[SUCCESS] ChromaDB successfully initialized with {len(unique_docs)} sections across Bearing & Motor domains.")
    return vector_db


# ============================================================
# TELEMETRY-GROUNDED LLM REASONING
# ============================================================

def format_telemetry_for_prompt(telemetry: Optional[Dict[str, Any]]) -> str:
    if not telemetry:
        return "No real-time physical telemetry available. Basing diagnosis on general operational thresholds."
    lines = []
    for k, v in telemetry.items():
        name = k.replace("_", " ").title()
        if isinstance(v, float):
            lines.append(f"- {name}: {v:.3f}")
        else:
            lines.append(f"- {name}: {v}")
    return "\n".join(lines)


def generate_llm_reasoning(fault_name, confidence_score, retrieved_context, telemetry=None, domain="bearing"):
    telemetry_str = format_telemetry_for_prompt(telemetry)

    prompt = f"""You are a Principal Industrial Reliability Engineer conducting an autonomous Root Cause Analysis (RCA).

DIAGNOSTIC FINDING:
Condition: {fault_name}
AI Confidence: {confidence_score:.2f}%
Domain Scope: {domain.upper()}

OBSERVED SENSOR TELEMETRY & PHYSICAL METRICS:
{telemetry_str}

ENGINEERING SOP KNOWLEDGE BASE:
\"\"\"
{retrieved_context}
\"\"\"

INSTRUCTIONS:
1. Ground your analysis strictly in the provided SOP and observed physical metrics.
2. Explicitly explain how the observed telemetry (e.g. vibration RMS, peak, temperature, thermal gradient, speed, load) supports the detected condition.
3. Provide the response in this exact structured format:

ROOT CAUSE:
<2-3 concise, highly technical sentences explaining the mechanical or electrical root cause, referencing specific telemetry values>

PRIMARY PHYSICAL SYMPTOMS:
1. <Symptom with measured telemetry value>
2. <Symptom with measured telemetry value>

IMMEDIATE SAFETY ACTIONS:
1. <Immediate action following plant LOTO / shutdown criteria>

CORRECTIVE MAINTENANCE PLAN:
1. <Step-by-step mechanical/electrical fix grounded in the SOP>
2. <Component replacement or tolerance restoration specification>

PREVENTIVE MAINTENANCE & MONITORING:
1. <Long-term condition monitoring interval and threshold to prevent recurrence>

Keep your output professional, decisive, engineering-focused, and free of conversational fluff."""

    try:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            # keep_alive keeps the model resident in RAM between calls so
            # Ollama doesn't have to cold-reload it from disk on every
            # request (a common source of extra 10-30s latency on
            # CPU-only machines when the default 5-minute idle unload
            # kicks in between demo runs).
            keep_alive="30m",
            # num_predict bounds how many tokens the model is allowed to
            # generate, so response time is capped and predictable
            # instead of varying with however verbose the model decides
            # to be. 320 tokens is enough for the 5-section structured
            # output requested above.
            options={
                "num_predict": 320,
                "temperature": 0.3,
                "num_thread": 8,   # explicitly use physical cores (Ryzen 7
                                    # 8840HS: 8 cores / 16 threads) — Ollama
                                    # sometimes under-uses available CPU
                                    # threads by default
            },
        )
        return response["response"].strip()
    except Exception as e:
        return f"[Ollama reasoning unavailable: {e}]\nFalling back to retrieved SOP engineering guidelines."


# ============================================================
# LANGGRAPH ADAPTIVE RAG STATE & NODES
# ============================================================

class RAGState(TypedDict):
    domain: str
    class_id: int
    fault_name: str
    fault_code: str
    confidence_score: float
    telemetry: Optional[Dict[str, Any]]
    query: str
    candidates: List[dict]
    retrieved_context: str
    vector_db: object


def _route_query_node(state: RAGState):
    domain = state.get("domain", "bearing")
    class_id = state["class_id"]
    confidence = state["confidence_score"]

    if domain == "motor":
        fault_code = MOTOR_CODE_MAPPING.get(class_id, "MOTOR_NORMAL")
        fault_name = MOTOR_CLASS_MAPPING.get(class_id, "Normal Motor Operation")
    else:
        fault_code = BEARING_CODE_MAPPING.get(class_id, "NORMAL")
        fault_name = BEARING_CLASS_MAPPING.get(class_id, "Normal Operating Condition")

    state["fault_code"] = fault_code
    state["fault_name"] = fault_name

    # Adaptive query tuning
    if class_id == 0:
        query = f"{fault_code} {fault_name} routine maintenance baseline limits thresholds inspection"
    elif confidence >= 85.0:
        query = f"{fault_code} {fault_name} root cause inspection procedure corrective maintenance tolerance"
    else:
        query = f"{fault_code} {fault_name} diagnostic verification possible causes vibration thermal inspection"

    state["query"] = query
    return state


def _retrieve_node(state: RAGState):
    vector_db = state["vector_db"]
    query = state["query"]
    fault_code = state["fault_code"]
    domain = state.get("domain", "bearing")
    candidates = []

    if vector_db:
        try:
            # Metadata filter by fault_code
            results = vector_db.similarity_search(
                query, k=5, filter={"fault_code": fault_code}
            )
            for doc in results:
                txt = doc.page_content.strip()
                candidates.append({"text": txt, "fault_code": fault_code})
        except Exception as e:
            # Fallback
            results = vector_db.similarity_search(query, k=6)
            for doc in results:
                if doc.metadata.get("fault_code") == fault_code:
                    candidates.append({"text": doc.page_content.strip(), "fault_code": fault_code})

    state["candidates"] = candidates
    return state


def _rerank_node(state: RAGState):
    candidates = state["candidates"]
    if not candidates:
        state["retrieved_context"] = "No specific SOP engineering knowledge retrieved for this fault code."
        return state

    docs = [c["text"] for c in candidates]
    if cohere_client is not None:
        try:
            rerank_result = cohere_client.rerank(
                model="rerank-english-v3.0",
                query=state["query"],
                documents=docs,
                top_n=min(3, len(docs))
            )
            top_indices = [r.index for r in rerank_result.results]
            selected = [docs[i] for i in top_indices]
            state["retrieved_context"] = "\n\n---\n\n".join(selected)
            return state
        except Exception:
            pass

    # Default top 3
    selected = docs[:3]
    state["retrieved_context"] = "\n\n---\n\n".join(selected)
    return state


def _build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("route_query", _route_query_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("rerank", _rerank_node)

    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", END)
    return graph.compile()


_rag_graph = _build_rag_graph()


# ============================================================
# MAIN ENTRY POINT: RCA REPORT GENERATION
# ============================================================

def generate_rca_report(class_id, confidence_score, vector_db=None, domain="bearing", telemetry=None):
    """
    Executes LangGraph Adaptive RAG and generates telemetry-grounded Root Cause Analysis.
    """
    if domain == "motor":
        fault_name = MOTOR_CLASS_MAPPING.get(class_id, "Normal Motor Operation")
        fault_code = MOTOR_CODE_MAPPING.get(class_id, "MOTOR_NORMAL")
    else:
        fault_name = BEARING_CLASS_MAPPING.get(class_id, "Normal Operating Condition")
        fault_code = BEARING_CODE_MAPPING.get(class_id, "NORMAL")

    if vector_db is None:
        embeddings = get_embeddings()
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        else:
            vector_db = build_vector_store()

    if vector_db is None:
        return "RAG ERROR: ChromaDB Vector Database could not be initialized."

    initial_state: RAGState = {
        "domain": domain,
        "class_id": class_id,
        "fault_name": fault_name,
        "fault_code": fault_code,
        "confidence_score": confidence_score,
        "telemetry": telemetry,
        "query": "",
        "candidates": [],
        "retrieved_context": "",
        "vector_db": vector_db
    }

    final_state = _rag_graph.invoke(initial_state)
    retrieved_context = final_state["retrieved_context"]

    # Generate LLM reasoning grounded in physical telemetry
    llm_reasoning = generate_llm_reasoning(
        fault_name=fault_name,
        confidence_score=confidence_score,
        retrieved_context=retrieved_context,
        telemetry=telemetry,
        domain=domain
    )

    reranker_status = "Cohere Rerank v3 (Active)" if cohere_client else "Strict Metadata Filtering (Active)"

    report = f"""==================================================================
        AUTONOMOUS INDUSTRIAL ROOT CAUSE ANALYSIS (RCA)
==================================================================

Target Domain    : {domain.upper()} CONDITION MONITORING
Detected Status  : {fault_name}
Fault Code       : {fault_code}
AI Confidence    : {confidence_score:.2f}%
Retrieval Mode   : LangGraph Adaptive Router | {reranker_status}

------------------------------------------------------------------
ENGINEERING STANDARD OPERATING PROCEDURE (SOP) CONTEXT
------------------------------------------------------------------
{retrieved_context}

------------------------------------------------------------------
PHYSICAL TELEMETRY-GROUNDED AI REASONING ({LLM_MODEL})
------------------------------------------------------------------
{llm_reasoning}

==================================================================
"""
    return report


if __name__ == "__main__":
    vdb = build_vector_store()
    # Test Bearing RCA
    print("\n--- Testing Bearing RCA ---")
    b_telemetry = {"vibration_rms": 0.942, "peak_vibration": 3.84, "motor_speed_rpm": 1750, "bearing_temp_c": 64.5}
    b_report = generate_rca_report(class_id=1, confidence_score=98.5, vector_db=vdb, domain="bearing", telemetry=b_telemetry)
    print(b_report[:500] + "...")

    # Test Motor RCA
    print("\n--- Testing Motor RCA ---")
    m_telemetry = {"t_max": 74.2, "t_mean": 48.5, "asymmetry_index": 1.48, "load_current_A": 4.0}
    m_report = generate_rca_report(class_id=1, confidence_score=94.1, vector_db=vdb, domain="motor", telemetry=m_telemetry)
    print(m_report[:500] + "...")