import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample

fold_csv_path = '/media/data3/users/loctx/REFERENCE_5fold.csv'
mat_folder = '/media/data3/users/loctx/Training_WFDB'
val_ratio = 0.1
target_sampling_rate = 100
target_duration = 10
selected_test_fold = 4  # choose fold test: 0, 1, 2, 3, 4
num_classes = 9


def resample_ecg(data, original_freq=500, target_freq=100, duration=10):
    num_samples = int(duration * target_freq)
    return resample(data, num_samples, axis=1)


def standardize_length(data, target_length_sec, sampling_rate):
    target_len = target_length_sec * sampling_rate
    standardized = []
    for lead in data:
        if lead.shape[0] > target_len:
            standardized.append(lead[:target_len])
        elif lead.shape[0] < target_len:
            repeated = np.tile(lead, (target_len // lead.shape[0] + 1))[:target_len]
            standardized.append(repeated)
        else:
            standardized.append(lead)
    return np.array(standardized)


def load_and_process_from_folds(csv_path, mat_folder, test_fold=0,
                                 target_sampling_rate=100, target_duration=10):
    df = pd.read_csv(csv_path)
    all_data = []
    all_labels = []
    all_folds = []

    for _, row in df.iterrows():
        record_id = row['recording']
        labels_raw = [row["First_label"], row["Second_label"], row["Third_label"]]
        label_list = []

        for lbl in labels_raw:
            if pd.notna(lbl):
                try:
                    lbl_int = int(lbl)
                    if 1 <= lbl_int <= num_classes:
                        label_list.append(lbl_int)
                except (ValueError, TypeError):
                    continue

        if len(label_list) == 0:
            continue

        fold = row["fold"]
        mat_path = os.path.join(mat_folder, record_id + ".mat")
        if not os.path.exists(mat_path):
            print(f"[Warning] Missing file: {mat_path}")
            continue

        mat_data = sio.loadmat(mat_path)
        if "val" not in mat_data:
            print(f"[Warning] 'val' key not found in {mat_path}")
            continue

        ecg = mat_data["val"]  # shape (12, N)

        ecg = resample_ecg(ecg, 500, target_sampling_rate, target_duration)
        ecg = standardize_length(ecg, target_duration, target_sampling_rate)

        # multi-hot vector
        label_vector = np.zeros(num_classes)
        for lbl in label_list:
            label_vector[lbl - 1] = 1

        all_data.append(ecg)
        all_labels.append(label_vector)
        all_folds.append(fold)

    return np.array(all_data), np.array(all_labels), np.array(all_folds)


def split_train_val_test(data, labels, folds, test_fold, val_ratio=0.1):
    test_idx = np.where(folds == test_fold)[0]
    train_val_idx = np.where(folds != test_fold)[0]

    x_test, y_test = data[test_idx], labels[test_idx]
    x_train_val, y_train_val = data[train_val_idx], labels[train_val_idx]

    perm = np.random.permutation(len(x_train_val))
    x_train_val = x_train_val[perm]
    y_train_val = y_train_val[perm]

    n_val = int(len(x_train_val) * val_ratio)
    x_val, y_val = x_train_val[:n_val], y_train_val[:n_val]
    x_train, y_train = x_train_val[n_val:], y_train_val[n_val:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def main():
    data, labels, folds = load_and_process_from_folds(
        fold_csv_path, mat_folder, test_fold=selected_test_fold
    )

    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(
        data, labels, folds, test_fold=selected_test_fold, val_ratio=val_ratio
    )

    print(f"Train: {x_train.shape}, {y_train.shape}")
    print(f"Val:   {x_val.shape}, {y_val.shape}")
    print(f"Test:  {x_test.shape}, {y_test.shape}")

    save_dir = f"/media/data3/users/loctx/MRM/Fold{selected_test_fold}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "x_train.npy"), x_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "x_val.npy"), x_val)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "x_test.npy"), x_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)


if __name__ == "__main__":
    main()
