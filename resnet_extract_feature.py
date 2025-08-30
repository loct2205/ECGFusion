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
    def __init__(self, mat_dir, label_path, file_list, csv_file,
                 window_size=6*500, img_size=224, is_test=False,
                 noise_level=0.015, scale_range=0.15, shift_max=15, seed=42,
                 lead_dropout_prob=0.3, max_leads_to_drop=3):
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
        label_vector = torch.zeros(self.num_classes)
        for lbl in label_list:
            if not pd.isna(lbl) and lbl != -1:
                label_vector[int(lbl) - 1] = 1

        if not self.is_test_set:
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
def get_dataloaders(mat_dir, label_path, train_files, test_files, batch_size=32):
    train_dataset = ECGDataset(mat_dir, label_path, train_files)
    #val_dataset = ECGDataset(mat_dir, label_path, val_files)
    test_dataset = ECGDataset(mat_dir, label_path, test_files, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

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
    def __init__(self, block=ResidualBlock1D, layers=[2, 2, 2, 2], num_classes=9, input_channels=12,
                 attention_heads=8, dropout_prob=0.5):
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



    def extract_features(self, x):  # x: (B, 12, 3, 224, 224)
        B, L, C, H, W = x.shape
        assert L == self.num_leads, f"Expected {self.num_leads} leads, got {L}"

        # Merge batch and lead dimensions
        x = x.view(B * L, C, H, W)  # (B*12, 3, 224, 224)

        # Extract embeddings using Swin (using SwinModel, not classifier)
        output = self.backbone(x)
        features = output.last_hidden_state  # (B*12, seq_len, hidden_size)

        # Debugging
        #print(f"Features shape after SwinModel: {features.shape}")  # (B*12, seq_len, hidden_size)

        # Split back into (B, L, seq_len, hidden_size)
        features = features.view(B, L, 49, self.hidden_size)  # (B, 12, 49, hidden_size)

        # Apply attention over leads (for each batch independently)
        attended = []
        for i in range(B):
            batch_leads = features[i]  # (L, seq_len, hidden_size) for this batch

            # Apply attention on the leads of this batch only
            attended_leads = self.lead_attention(batch_leads)  # (L, seq_len, hidden_size)

            attended.append(attended_leads)

        # Stack attended leads for all batches
        attended = torch.stack(attended, dim=0)  # (B, L, seq_len, hidden_size)

        # Fuse leads
        fused = attended.mean(dim=1)  # (B, seq_len, hidden_size)
        return fused # Feature

    def forward(self, x):
        fused = self.extract_features(x)  # (B, seq_len, hidden_size)
        # Classify
        logits = self.classifier(fused.mean(dim=1))  # (B, num_classes)
        return logits

def extract_and_save_features(model, dataloader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    file_feature_dict = {}

    with torch.no_grad():
        for x_batch, y_batch, file_ids in tqdm(dataloader, desc="Extracting features"):
            x_batch = x_batch.to(device)  # shape: (B, 12, T)
            features = model(x_batch, extract_features=True).cpu()  # shape: (B, 512)

            for i in range(x_batch.size(0)):
                file_id = file_ids[i]
                feat = features[i]           # shape: (512,)
                label = y_batch[i].cpu()     # shape: (num_classes,)

                if file_id not in file_feature_dict:
                    file_feature_dict[file_id] = {'features': [], 'labels': []}

                file_feature_dict[file_id]['features'].append(feat)
                file_feature_dict[file_id]['labels'].append(label)

    # Save each file_id's features and labels
    for file_id, data in file_feature_dict.items():
        try:
            feats_tensor = torch.stack(data['features'])    # (N, 512)
            labels_tensor = torch.stack(data['labels'])     # (N, num_classes)

            torch.save({
                'features': feats_tensor,
                'labels': labels_tensor
            }, os.path.join(save_dir, f"{file_id}.pt"))
        except Exception as e:
            print(f"Failed to save {file_id}: {e}")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data's dir
mat_dir = "./Training_WFDB"
label_path = "./REFERENCE_multi_label.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 9

for fold in range(5):
    df = pd.read_csv('./KFold/REFERENCE_5fold.csv')
    print(fold)
    print(f"\n----- Fold {fold} -----")

    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]

    print(f"train: {len(train_df)}, validation: {len(val_df)}")

    train_files = train_df['recording'].values
    train_labels = train_df['First_label'].values

    test_files = val_df['recording'].values
    test_labels = val_df['First_label'].values

    print(f"Train: {len(train_files)}, Test: {len(test_files)}")
    ########
    feature_dir = f'./KFold/Fold{fold}/fearure'
    csv_file = f'./KFold/Fold{fold}/fearure/log.csv'

    # Initializing model and load state
    model = ResNet1DWithAttention(num_classes=num_classes).to(device)
    model.to(device)
    model.load_state_dict(torch.load(f"./KFold/Fold{fold}/resnet1d_attention.pth", map_location=device))
    # load data
    train_loader, test_loader = get_dataloaders(mat_dir, label_path, train_files, test_files, csv_file)

    # extract feature and save
    extract_and_save_features(model, test_loader, device, feature_dir)
    extract_and_save_features(model, train_loader, device, feature_dir)


