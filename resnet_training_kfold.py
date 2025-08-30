import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, recall_score
from torch.nn.functional import sigmoid
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.models as models
from transformers import SwinForImageClassification, SwinModel, AutoImageProcessor
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import random


class ECGDataset(Dataset):
    def __init__(self, mat_dir, label_path, file_list, csv_file="./KFold/Resnet/Fold4/log.csv",
                 window_size=6*500, img_size=224, is_test=False,
                 noise_level=0.0, scale_range=0.05, shift_max=0, seed=42,
                 lead_dropout_prob=0.0, max_leads_to_drop=1):
        self.mat_dir = mat_dir
        self.window_size = window_size
        self.img_size = img_size
        self.is_test = is_test
        self.csv_file = csv_file
        self.seed = seed
        self.py_rng = random.Random(seed)
        # Augmentation parameters
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.shift_max = shift_max
        self.lead_dropout_prob = lead_dropout_prob
        self.max_leads_to_drop = max_leads_to_drop

        # Load label file
        df_labels = pd.read_csv(label_path)
        self.label_dict = {}
        for _, row in df_labels.iterrows():
            labels = [row["First_label"], row["Second_label"], row["Third_label"]]
            self.label_dict[row["recording"]] = tuple([lbl if not np.isnan(lbl) else -1 for lbl in labels])

        # Stride values by first label
        self.offset_dict = {1: 353, 2: 412, 3: 250, 4: 77, 5: 619, 6: 278, 7: 341, 8: 320, 9: 86}

        self.samples = []
        log_data = []

        for mat_file in file_list:
            mat_path = os.path.join(mat_dir, mat_file)
            try:
                mat_data = loadmat(mat_path)
                val_data = mat_data.get('val', None)
                if val_data is None or not isinstance(val_data, np.ndarray):
                    continue
            except:
                continue

            file_id = mat_file.replace(".mat", "")
            label_group = self.label_dict.get(file_id, None)
            if label_group is None:
                continue

            first_label = label_group[0]
            if not self.is_test and first_label not in self.offset_dict:
                print(f"Skipping file {mat_file} - Label group {label_group} not in offset_dict.")
                continue
            # Test's stride = 77, train/val uses offset_dict
            stride = 77 if self.is_test else self.offset_dict.get(first_label)
            signal_length = val_data.shape[1]

            if signal_length < self.window_size or stride is None:
                continue

            start_indices = np.arange(0, signal_length - self.window_size + 1, stride)
            if len(start_indices) == 0:
                continue

            final_segments_shape = (len(start_indices), val_data.shape[0], self.window_size)
            log_data.append([mat_file, signal_length, label_group, stride, final_segments_shape])

            for start_idx in start_indices:
                self.samples.append((mat_path, start_idx, label_group))

        # Save log
        df_log = pd.DataFrame(log_data, columns=["mat_file", "signal_length", "label_group", "stride", "final_segments_shape"])
        df_log.to_csv(self.csv_file, index=False)

    def __len__(self):
        return len(self.samples)

    def _add_noise(self, segment):
        noise = np.random.normal(0, self.noise_level * np.std(segment), segment.shape)
        return segment + noise

    def _scale_amplitude(self, segment):
        scale_factor = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)
        return segment * scale_factor

    def _time_shift(self, segment):
        shift_amount = np.random.randint(-self.shift_max, self.shift_max + 1)
        shifted_segment = np.roll(segment, shift_amount, axis=-1)
        # Zero-pad the space opened by roll
        if shift_amount > 0:
            shifted_segment[:, :shift_amount] = 0
        elif shift_amount < 0:
            shifted_segment[:, shift_amount:] = 0
        return shifted_segment

    def _lead_dropout(self, segment):
        num_leads = segment.shape[0]
        # Ensure not dropping all leads if max_leads_to_drop is high relative to num_leads
        max_drop = min(self.max_leads_to_drop, num_leads - 1 if num_leads > 1 else 0)
        if max_drop <= 0: return segment

        num_to_drop = np.random.randint(1, max_drop + 1)

        leads_to_drop_indices = np.random.choice(num_leads, num_to_drop, replace=False)
        segment[leads_to_drop_indices, :] = 0
        return segment

    def __getitem__(self, idx):
        mat_path, start_idx, _ = self.samples[idx]
        file_id = os.path.basename(mat_path).replace(".mat", "")
        val_data = loadmat(mat_path)['val']

        segment = np.stack([
            val_data[lead_idx, start_idx:start_idx + self.window_size]
            for lead_idx in range(val_data.shape[0])
        ]).astype(np.float32)

        label_list = self.label_dict[file_id]
        label_vector = torch.zeros(num_classes)
        for lbl in label_list:
            if not pd.isna(lbl) and lbl != -1:
                label_vector[int(lbl) - 1] = 1

        if not self.is_test:
            seg = segment.copy()
            if self.lead_dropout_prob > 0 and self.py_rng.random() < self.lead_dropout_prob:
                seg = self._lead_dropout(seg)
            if self.noise_level > 0 and self.py_rng.random() < 0.7:
                seg = self._add_noise(seg)
            if self.scale_range > 0 and self.py_rng.random() < 0.7:
                seg = self._scale_amplitude(seg)
            if self.shift_max > 0 and self.py_rng.random() < 0.7:
                seg = self._time_shift(seg)
            segment = seg

        return torch.tensor(segment, dtype=torch.float32), label_vector, file_id


# DataLoader function
def get_dataloaders(mat_dir, label_path, train_files, val_files, test_files, batch_size=32):
    train_dataset = ECGDataset(mat_dir, label_path, train_files)
    val_dataset = ECGDataset(mat_dir, label_path, val_files, is_test=True)
    test_dataset = ECGDataset(mat_dir, label_path, test_files, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class MultiHeadSelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention1D, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x): # x shape: (batch_size, channels(embed_dim), sequence_length)
        x_permuted = x.permute(0, 2, 1) # (batch_size, sequence_length, channels(embed_dim)) for MHA

        # Self-attention
        attn_output, _ = self.mha(x_permuted, x_permuted, x_permuted)

        # Add and Norm
        x_after_mha_res = x_permuted + attn_output
        x_normed1 = self.norm(x_after_mha_res)

        # Feed-forward
        ff_output = self.feed_forward(x_normed1)

        # Add and Norm
        x_after_ff_res = x_normed1 + ff_output
        x_normed2 = self.norm2(x_after_ff_res)

        x_out = x_normed2.permute(0, 2, 1) #(batch_size, channels(embed_dim), sequence_length)
        return x_out


class ResNet1DWithAttention(nn.Module):
    def __init__(self, block=ResidualBlock1D, layers=[1, 1, 1, 1], num_classes=9, input_channels=12,
                 attention_heads=8, dropout_prob=0.4):
        super(ResNet1DWithAttention, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv1d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # Output channels: 512

        # Attention layer
        # embed_dim for attention should match output channels of the last ResNet layer (512)
        self.attention = MultiHeadSelfAttention1D(embed_dim=512, num_heads=attention_heads,
                                                  dropout=0.1)  # Dropout within attention

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout before final FC layer
        self.fc = nn.Linear(512, num_classes)  # Final FC layer

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layer_blocks = []
        layer_blocks.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layer_blocks.append(block(out_channels, out_channels))

        return nn.Sequential(*layer_blocks)

    def forward(self, x, extract_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  #(batch_size, 512, seq_len_after_res_layers)

        x = self.attention(x) #(batch_size, 512, seq_len_after_attention)

        x = self.avgpool(x)  #(batch_size, 512, 1)
        x = torch.flatten(x, 1)  # Shape: (batch_size, 512)

        if extract_features:
            return x  # Return 512-dimensional feature vector

        x = self.dropout(x)
        x = self.fc(x)
        return x

# --- Training with Early Stopping ---
def train_model(model, train_loader, test_loader, num_classes, device,
                save_path="./KFold/Resnet/Fold4/resnet1d_attention.pth",
                num_epochs=42, learning_rate=0.001, weight_decay=5e-4,
                patience=10, threshold=0.5):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = None

    print("===== Begin Training =====")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for x_batch, y_batch, _ in loop:
            if x_batch.size(0) == 0:
                continue

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            preds = (torch.sigmoid(outputs) >= threshold).float()
            acc = (preds == y_batch).float().mean(dim=1).mean()  # Accuracy per sample

            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size

            loop.set_postfix(train_loss=total_loss / total_samples,
                             train_acc=total_acc / total_samples)

        avg_train_loss = total_loss / total_samples
        avg_train_acc = total_acc / total_samples
        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")

        # ===== Validation =====
        avg_val_loss, avg_val_acc = validate_model(model, val_loader, criterion, device, threshold)
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # ===== Early Stopping =====
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} → {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            try:
                torch.save(model.state_dict(), save_path)
                best_model_path = save_path
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s). (Patience = {patience})")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        scheduler.step()

    print("===== Training Completed =====")
    return best_model_path
def validate_model(model, val_loader, criterion, device, threshold=0.5):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_samples = 0

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Validation", leave=False)
        for x_batch, y_batch, _ in val_loop:
            if x_batch.size(0) == 0:
                continue

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            preds = (torch.sigmoid(outputs) >= threshold).float()
            acc = (preds == y_batch).float().mean(dim=1).mean()  # Accuracy per sample

            batch_size = x_batch.size(0)
            val_loss += loss.item() * batch_size
            val_acc += acc.item() * batch_size
            val_samples += batch_size

            val_loop.set_postfix(val_loss=val_loss / val_samples,
                                 val_acc=val_acc / val_samples)

    avg_val_loss = val_loss / val_samples
    avg_val_acc = val_acc / val_samples
    return avg_val_loss, avg_val_acc

def evaluate_voting(model, test_loader, device, model_name, threshold=0.5):
    model.eval()
    frame_predictions = {}
    frame_probabilities = {}

    with torch.no_grad():
        for x_batch, y_batch, file_id_batch in test_loader:
            x_batch = x_batch.to(device)
            with autocast("cuda"):
                outputs = model(x_batch)  # (B, num_classes)
            probs = torch.sigmoid(outputs)  # (batch_size, num_classes)
            preds = (probs > 0.5).float()

            for i in range(len(y_batch)):
                file_id = file_id_batch[i]  # ID of file ECG
                if file_id not in frame_predictions:
                    frame_predictions[file_id] = []
                    frame_probabilities[file_id] = []

                frame_predictions[file_id].append(preds[i].cpu().numpy())
                frame_probabilities[file_id].append(probs[i].cpu().numpy())

    majority_voting, mean_voting, median_voting, mean_probs_dict, median_probs_dict = {}, {}, {}, {}, {}

    for file_id in frame_predictions:
        #print(file_id)
        votes = np.array(frame_predictions[file_id])
        probs = np.array(frame_probabilities[file_id])

        # Majority Voting
        majority_label = (np.sum(votes, axis=0) >= (votes.shape[0] / 2)).astype(np.float32)
        majority_voting[file_id] = majority_label

        # Mean Voting
        mean_probs = np.mean(probs, axis=0)
        mean_label = (mean_probs > threshold).astype(np.float32)
        mean_voting[file_id] = mean_label
        mean_probs_dict[file_id] = mean_probs

        # Median Voting
        median_probs = np.median(probs, axis=0)
        median_label = (median_probs > threshold).astype(np.float32)
        median_voting[file_id] = median_label
        median_probs_dict[file_id] = median_probs

    # Save results
    df = pd.DataFrame({
        "File ID": list(majority_voting.keys()),
        "Majority Voting": list(majority_voting.values()),
        "Mean Voting": list(mean_voting.values()),
        "Median Voting": list(median_voting.values())
    })
    df.to_csv(f"./KFold/Resnet/Fold4/{model_name}_voting_results.csv", index=False)

    # Compute evaluation metrics
    y_true_dict = {}
    for _, y_batch, file_id_batch in test_loader:
        for i in range(len(y_batch)):
            file_id = file_id_batch[i]
            if file_id not in y_true_dict:
                y_true_dict[file_id] = y_batch[i].cpu().numpy()
    print('continue')
    y_true = np.array(list(y_true_dict.values()))
    y_mean = np.array(list(mean_voting.values()))
    y_median = np.array(list(median_voting.values()))

    y_mean_probs = np.array(list(mean_probs_dict.values()))
    y_median_probs = np.array(list(median_probs_dict.values()))

    if np.isnan(y_mean_probs).any():
        nan_indices = np.where(np.isnan(y_mean_probs))
        print("[!] NaN detected in y_mean_probs at indices:", nan_indices)

        # In thông tin file gây lỗi
        file_ids = list(mean_probs_dict.keys())
        for i in np.unique(nan_indices[0]):
            print(f"File with NaN probs: {file_ids[i]}")
            print("Probs:", mean_probs_dict[file_ids[i]])
    auc_mean = roc_auc_score(y_true, y_mean_probs, average="macro")
    recall_mean = recall_score(y_true, y_mean, average="macro", zero_division=0)
    # accuracy_mean = accuracy_score(y_true, y_mean)
    accuracy_mean = (y_true == y_mean).astype(float).mean()
    f1_mean = f1_score(y_true, y_mean, average="macro", zero_division=0)

    auc_median = roc_auc_score(y_true, y_median_probs, average="macro")
    recall_median = recall_score(y_true, y_median, average="macro", zero_division=0)
    #accuracy_median = accuracy_score(y_true, y_median)
    accuracy_median = (y_true == y_median).astype(float).mean()
    f1_median = f1_score(y_true, y_median, average="macro", zero_division=0)

    with open(f"./KFold/Resnet/Fold4/{model_name}_voting_results.txt", "w") as f:
        f.write("Mean Voting:\n")
        f.write(f"  Accuracy: {accuracy_mean:.4f}\n")
        f.write(f"  F1-score: {f1_mean:.4f}\n")
        f.write(f"  Recall: {recall_mean:.4f}\n")
        f.write(f"  AUC: {auc_mean:.4f}\n\n")

        f.write("Median Voting:\n")
        f.write(f"  Accuracy: {accuracy_median:.4f}\n")
        f.write(f"  F1-score: {f1_median:.4f}\n")
        f.write(f"  Recall: {recall_median:.4f}\n")
        f.write(f"  AUC: {auc_median:.4f}\n")

    print(f"Results saved in {model_name}_voting_results.txt")
    # confusion matrix
    # multi-label to single-label
    y_true_single = np.argmax(y_true, axis=1)
    y_mean_single = np.argmax(y_mean_probs, axis=1)
    y_median_single = np.argmax(y_median_probs, axis=1)
    # plot_confusion_matrix(y_true_single, y_mean_single,
    #                       f"{model_name} - Mean Voting - Confusion Matrix",
    #                       f"{model_name}_mean_confusion_matrix.png")
    #
    # plot_confusion_matrix(y_true_single, y_median_single,
    #                       f"{model_name} - Median Voting - Confusion Matrix",
    #                       f"{model_name}_median_confusion_matrix.png")
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()

# Data's dir
mat_dir = "./Training_WFDB"
label_path = "./REFERENCE_multi_label.csv"

df = pd.read_csv('./KFold/REFERENCE_5fold.csv')
fold = 4
print(fold)
print(f"\n----- Fold {fold} -----")

test_df = df[df['fold'] == fold]
train_val_df = df[df['fold'] != fold]
train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

train_files = train_df['recording'].values
val_files = val_df['recording'].values
test_files = test_df['recording'].values

train_labels = train_df['First_label'].values
val_labels = val_df['First_label'].values
test_labels = test_df['First_label'].values

train_loader, val_loader, test_loader = get_dataloaders(mat_dir, label_path, train_files, val_files, test_files)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initializing model
num_classes = 9
model = ResNet1DWithAttention(num_classes=num_classes).to(device)
#####################################
# Train
train_model(model, train_loader, val_loader, num_classes, device)
model.load_state_dict(torch.load(f"./KFold/Resnet/Fold4/resnet1d_attention.pth", map_location=device))
model_name = 'resnet1d_attention'
evaluate_voting(model, test_loader, device, model_name, threshold=0.5)