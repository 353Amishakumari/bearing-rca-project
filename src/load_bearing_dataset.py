## Raw .mat files,Load vibration signal,Split into 1024 samples,Assign label,Assign group ID,Save X, y, groups.
import os # Python module used to work with files, folders, and file paths.
import glob   # find files in a folder using a pattern.
import numpy as np
import scipy.io as sio # used to read MATLAB .mat files.

# Dataset Paths
RAW_DIR = os.path.join("dataset", "raw")
PROCESSED_DIR = os.path.join("dataset", "processed")

# Class Labels Mapping based on file names
LABEL_MAP = {
    "Normal": 0,  # Healthy Bearing
    "IR": 1,      # Inner Race Fault
    "B": 2,       # Ball Fault
    "OR": 3       # Outer Race Fault
}

SEGMENT_LENGTH = 1024  # Time-series window size

def get_label_from_filename(filename):
    basename = os.path.basename(filename)
    for key in LABEL_MAP:
        if key in basename:
            return LABEL_MAP[key]
    return 0

def process_mat_files():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    mat_files = glob.glob(os.path.join(RAW_DIR, "*.mat"))

    if not mat_files:
        print(f"❌ No .mat files found in {RAW_DIR}! Please add CWRU dataset files.")
        return

    all_segments = []
    all_labels = []
    all_groups = []  # NEW: tracks which source file each segment came from

    print("🔄 Processing .mat files...")
    for file_idx, file_path in enumerate(mat_files):
        label = get_label_from_filename(file_path)
        mat_data = sio.loadmat(file_path)

        de_key = [k for k in mat_data.keys() if 'DE_time' in k]
        if not de_key:
            de_key = [k for k in mat_data.keys() if not k.startswith('__')]

        if de_key:
            signal = mat_data[de_key[0]].flatten()
            num_segments = len(signal) // SEGMENT_LENGTH

            for i in range(num_segments):
                seg = signal[i * SEGMENT_LENGTH: (i + 1) * SEGMENT_LENGTH]
                all_segments.append(seg)
                all_labels.append(label)
                all_groups.append(file_idx)  # NEW: same file = same group ID

            print(f"  ✅ Loaded: {os.path.basename(file_path)} | Segments: {num_segments} | Label: {label} | Group: {file_idx}")

    X = np.array(all_segments)
    y = np.array(all_labels)
    groups = np.array(all_groups)

    np.save(os.path.join(PROCESSED_DIR, "X_segments.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y_labels.npy"), y)
    np.save(os.path.join(PROCESSED_DIR, "groups.npy"), groups)  # NEW

    print(f"\n🎉 Process Complete! Dataset saved to '{PROCESSED_DIR}'")
    print(f"   X_segments shape: {X.shape}")# actual vibration data for all segement
    print(f"   y_labels shape:   {y.shape}") # the fault label for each segment
    print(f"   groups shape:     {groups.shape}") # which source file segment came from

if __name__ == "__main__":
    process_mat_files()  