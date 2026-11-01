import os
import glob
import numpy as np
import scipy.io as sio
from scipy.stats import kurtosis


# ============================================================
# PATHS
# ============================================================

CWRU_DIR = os.path.join("dataset", "raw")
PADERBORN_DIR = os.path.join(
    "dataset",
    "cross_dataset",
    "paderborn"
)


# ============================================================
# CWRU SIGNAL LOADER
# ============================================================

def load_cwru_signal():

    files = glob.glob(
        os.path.join(CWRU_DIR, "*.mat")
    )

    if not files:
        raise FileNotFoundError(
            f"No CWRU .mat files found in {CWRU_DIR}"
        )

    # Use first file only for representative comparison
    file_path = files[0]

    mat_data = sio.loadmat(file_path)

    # Find Drive End vibration signal
    de_keys = [
        key for key in mat_data.keys()
        if "DE_time" in key
    ]

    if not de_keys:
        raise ValueError(
            f"DE_time signal not found in "
            f"{os.path.basename(file_path)}"
        )

    signal = np.asarray(
        mat_data[de_keys[0]]
    ).flatten()

    signal = signal.astype(np.float64)

    print()
    print("=" * 60)
    print("CWRU SIGNAL")
    print("=" * 60)
    print("File        :", os.path.basename(file_path))
    print("Signal key  :", de_keys[0])
    print("Signal length:", len(signal))

    return signal


# ============================================================
# PADERBORN SIGNAL EXTRACTION
# ============================================================

def extract_paderborn_signal(mat_path):

    mat_data = sio.loadmat(
        mat_path,
        squeeze_me=False
    )

    # Remove MATLAB internal keys
    main_keys = [
        key for key in mat_data.keys()
        if not key.startswith("__")
    ]

    if not main_keys:
        return None

    # Paderborn file normally contains one main structure
    root = mat_data[main_keys[0]]

    try:

        # ----------------------------------------------------
        # Access X structure
        # ----------------------------------------------------

        X = root["X"][0, 0]

        print()
        print("DEBUG: X shape  :", X.shape)
        print("DEBUG: X fields :", X.dtype.names)

        # ----------------------------------------------------
        # Search all channels for Mech_4kHz
        # ----------------------------------------------------

        for channel_index in range(X.shape[1]):

            channel = X[0, channel_index]

            raster_value = channel["Raster"]

            raster_text = str(raster_value)

            print(
                f"DEBUG: Channel {channel_index} "
                f"Raster = {raster_text}"
            )

            # ------------------------------------------------
            # Find vibration channel
            # ------------------------------------------------

            if "Mech_4kHz" in raster_text:

                print(
                    f"DEBUG: Mech_4kHz found "
                    f"at channel {channel_index}"
                )

                data = channel["Data"]

                print(
                    "DEBUG: Original Data shape:",
                    data.shape
                )

                # Remove MATLAB singleton dimensions
                data = np.asarray(data).squeeze()

                print(
                    "DEBUG: After squeeze shape:",
                    data.shape
                )

                # ------------------------------------------------
                # Handle nested object arrays if present
                # ------------------------------------------------

                while (
                    isinstance(data, np.ndarray)
                    and data.dtype == object
                    and data.size == 1
                ):
                    data = data.item()
                    data = np.asarray(data).squeeze()

                # Convert to one-dimensional signal
                data = np.asarray(data).flatten()

                print(
                    "DEBUG: Final signal length:",
                    len(data)
                )

                if len(data) <= 1:
                    print(
                        "WARNING: Extracted signal "
                        "contains <= 1 sample."
                    )
                    return None

                return data.astype(np.float64)

        print(
            "ERROR: Mech_4kHz channel not found."
        )

        return None

    except Exception as e:

        print(
            "ERROR while extracting Paderborn signal:"
        )

        print(e)

        return None


# ============================================================
# PADERBORN SIGNAL LOADER
# ============================================================

def load_paderborn_signal():

    folders = [
        "K001",
        "KA04",
        "KI04"
    ]

    for folder in folders:

        folder_path = os.path.join(
            PADERBORN_DIR,
            folder
        )

        files = glob.glob(
            os.path.join(
                folder_path,
                "*.mat"
            )
        )

        if not files:
            print(
                f"No MAT files found in {folder_path}"
            )
            continue

        # First MAT file as representative sample
        file_path = files[0]

        print()
        print("=" * 60)
        print("PADERBORN SIGNAL")
        print("=" * 60)
        print("Folder      :", folder)
        print("File        :", os.path.basename(file_path))

        signal = extract_paderborn_signal(
            file_path
        )

        if signal is not None:

            print(
                "Channel     : Mech_4kHz"
            )

            print(
                "Signal length:",
                len(signal)
            )

            return signal

    raise FileNotFoundError(
        "Could not extract a valid "
        "Paderborn Mech_4kHz signal."
    )


# ============================================================
# RAW SIGNAL STATISTICS
# ============================================================

def calculate_statistics(
    signal,
    dataset_name
):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    stats = {
        "Dataset": dataset_name,
        "Length": len(signal),
        "Mean": np.mean(signal),
        "Std": np.std(signal),
        "RMS": rms,
        "Peak": np.max(np.abs(signal)),
        "Min": np.min(signal),
        "Max": np.max(signal),
        "Range": np.max(signal) - np.min(signal),
        "Kurtosis": kurtosis(
            signal,
            fisher=False
        )
    }

    return stats


# ============================================================
# NORMALIZED SIGNAL STATISTICS
# ============================================================

def calculate_normalized_statistics(
    signal,
    dataset_name
):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    mean = np.mean(signal)

    std = np.std(signal)

    # Avoid division by zero
    if std < 1e-12:
        std = 1e-12

    normalized = (
        signal - mean
    ) / std

    rms = np.sqrt(
        np.mean(normalized ** 2)
    )

    stats = {
        "Dataset": dataset_name + " (Normalized)",
        "Length": len(normalized),
        "Mean": np.mean(normalized),
        "Std": np.std(normalized),
        "RMS": rms,
        "Peak": np.max(np.abs(normalized)),
        "Min": np.min(normalized),
        "Max": np.max(normalized),
        "Range": (
            np.max(normalized)
            - np.min(normalized)
        ),
        "Kurtosis": kurtosis(
            normalized,
            fisher=False
        )
    }

    return stats


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(
    stats
):

    print()
    print(stats["Dataset"])
    print("-" * 45)

    for key, value in stats.items():

        if key == "Dataset":
            continue

        if isinstance(value, (float, np.floating)):

            print(
                f"{key:12}: {value:.6f}"
            )

        else:

            print(
                f"{key:12}: {value}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CWRU vs PADERBORN SIGNAL COMPARISON")
    print("=" * 60)

    # --------------------------------------------------------
    # Load signals
    # --------------------------------------------------------

    cwru_signal = load_cwru_signal()

    paderborn_signal = load_paderborn_signal()

    # --------------------------------------------------------
    # Raw statistics
    # --------------------------------------------------------

    cwru_raw = calculate_statistics(
        cwru_signal,
        "CWRU"
    )

    paderborn_raw = calculate_statistics(
        paderborn_signal,
        "Paderborn"
    )

    # --------------------------------------------------------
    # Normalized statistics
    # --------------------------------------------------------

    cwru_normalized = calculate_normalized_statistics(
        cwru_signal,
        "CWRU"
    )

    paderborn_normalized = calculate_normalized_statistics(
        paderborn_signal,
        "Paderborn"
    )

    # --------------------------------------------------------
    # Print raw statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RAW SIGNAL STATISTICS")
    print("=" * 60)

    print_statistics(cwru_raw)

    print_statistics(paderborn_raw)

    # --------------------------------------------------------
    # Print normalized statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("NORMALIZED SIGNAL STATISTICS")
    print("=" * 60)

    print_statistics(cwru_normalized)

    print_statistics(paderborn_normalized)

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)

    print()
    print(
        "This comparison uses:"
    )
    print(
        "CWRU      -> DE_time vibration signal"
    )
    print(
        "Paderborn -> Mech_4kHz vibration signal"
    )

    print()