#(12, 3000) ---2D FFT(Fast Fourier transforms) spectrum 224 -> Pretrained model swintransform
#Spectrum is processed each image (12, F, T)
# #and resize (1, 256, 256) -> 1 chanel and get three times like # (3, 256, 256) that is input of model
#nperseg=128, noverlap=64 -> nperseg=512, noverlap=504
# loop x_batch[i]  -> output[] -> attended.mean-> out ()
#12 images from 1 sample -> each image into model
# -> extract features -> concat → classify
#multi-class and multi-label
#(8,12,244,244) => (8*12,244,244)
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
from collections import Counter
import torchvision.models as models
from transformers import SwinForImageClassification, SwinModel, AutoImageProcessor
from tqdm import tqdm
import scipy
import cv2
from scipy.signal import spectrogram
from torch import amp
from torch.amp import GradScaler, autocast
class ECGDataset(Dataset):
    def __init__(self, mat_dir, label_path, file_list, csv_file, window_size=6*500, img_size=224, is_test=False,
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

        # Load label
        df_labels = pd.read_csv(label_path)
        self.label_dict = dict()
        for _, row in df_labels.iterrows():
            labels = [row["First_label"], row["Second_label"], row["Third_label"]]
            self.label_dict[row["recording"]] = tuple([lbl if not np.isnan(lbl) else -1 for lbl in labels])
        self.offset_dict = {1: 353, 2: 412, 3: 250, 4: 77, 5: 619, 6: 278, 7: 341, 8: 320, 9: 86}

        self.samples = []
        log_data = []
        for mat_file in file_list:
            mat_path = os.path.join(mat_dir, mat_file)
            try:
                mat_data = loadmat(mat_path)
                val_data = mat_data.get('val', None)
                if val_data is None:
                    continue
            except:
                continue

            file_id = mat_file.replace(".mat", "")
            label_group = self.label_dict.get(file_id, None)
            if label_group is None:
                continue
            first_label = label_group[0]
            if not self.is_test and first_label not in self.offset_dict:
                print(f"Skipping file {mat_file} - Label group {label_group} not found in offset_dict.")
                continue
            # Test's stride = 77, train/val uses offset_dict
            stride = 77 if self.is_test else self.offset_dict.get(label_group[0])

            signal_length = val_data.shape[1]
            start_indices = np.arange(0, signal_length - self.window_size + 1, stride)
            final_segments_shape = (len(start_indices), val_data.shape[0], self.window_size)

            #print(mat_file, ":", signal_length, " ", label_group, " ", stride, " ", final_segments_shape)
            log_data.append([mat_file, signal_length, label_group, stride, final_segments_shape])

            for start_idx in start_indices:
                self.samples.append((mat_path, start_idx, label_group))

        df = pd.DataFrame(log_data, columns=["mat_file", "signal_length", "label_group", "stride", "final_segments_shape"])
        df.to_csv(self.csv_file, index=False)

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
        mat_path, start_idx, label = self.samples[idx]
        file_id = os.path.basename(mat_path).replace(".mat", "")
        mat_data = loadmat(mat_path)
        val_data = mat_data['val']

        segment = np.stack([val_data[lead_idx, start_idx:start_idx + self.window_size] for lead_idx in range(val_data.shape[0])]) #(12, 3000)
        #label = label - 1  # Taking label to about [0, num_classes-1]
        # Convert label_group to a multi-hot vector
        label_list = self.label_dict[file_id]  # Tuple of 3 labels
        label_vector = torch.zeros(num_classes)
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
        # Spectrogram
        Sxx_all = []
        for lead in segment:
            f, t, Sxx = spectrogram(lead, fs=500, nperseg=512, noverlap=504)
            # Log transform
            Sxx_log = np.log1p(Sxx)
            # Nomalization
            Sxx_log = Sxx_log / (np.max(Sxx_log) + 1e-8)
            # Resize
            Sxx_resized = cv2.resize(Sxx_log, (self.img_size, self.img_size))
            Sxx_all.append(Sxx_resized) # (12, F, T)

        Sxx_tensor = torch.tensor(np.array(Sxx_all), dtype=torch.float32)  # (12, 224, 224)
        # RGB 3 chanels
        Sxx_tensor = Sxx_tensor.unsqueeze(1)  # (12, 1, 224, 224)
        Sxx_tensor = Sxx_tensor.repeat(1, 3, 1, 1)  # (12, 3, 224, 224)

        return Sxx_tensor, label_vector.clone().detach().float(), file_id

# DataLoader function
def get_dataloaders(mat_dir, label_path, train_files, test_files, csv_file, batch_size=10, img_size=224):
    train_dataset = ECGDataset(mat_dir, label_path, train_files, csv_file, img_size=img_size)
    #val_dataset = ECGDataset(mat_dir, label_path, val_files, img_size=img_size)
    test_dataset = ECGDataset(mat_dir, label_path, test_files, csv_file, img_size=img_size, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers= 4, pin_memory=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers= 4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers= 4, pin_memory=True)

    return train_loader, test_loader
# Swin Transformer
class SwinWithLeadAttention(nn.Module):
    def __init__(self, num_classes=9, num_leads=12):
        super().__init__()
        self.num_leads = num_leads
        self.num_classes = num_classes

        # Load pretrained Swin model (SwinModel, not classifier)
        self.backbone = SwinModel.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224",
            ignore_mismatched_sizes=True
        )

        self.hidden_size = self.backbone.config.hidden_size

        # Attention over leads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=4, batch_first=True
        )
        self.lead_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Final classifier
        self.classifier = nn.Linear(self.hidden_size, num_classes)


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

def extract_and_save_features(model, dataloader, device, save_dir): # per sample grouped by file id
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    file_feature_dict = {}

    with torch.no_grad():
        for x_batch, y_batch, file_ids in tqdm(dataloader, desc="Extracting features"):
            x_batch = x_batch.to(device)  # (B, 12, 3, 224, 224)
            features = model.extract_features(x_batch)  # (B, seq_len, hidden_size)

            for i in range(x_batch.size(0)):
                file_id = file_ids[i]
                feat = features[i].cpu()  # shape (seq_len, hidden_size)
                label = y_batch[i].cpu()

                if file_id not in file_feature_dict:
                    file_feature_dict[file_id] = {'features': [], 'labels': []}

                file_feature_dict[file_id]['features'].append(feat)
                file_feature_dict[file_id]['labels'].append(label)

    # Save each file_id's collected features and labels
    for file_id, data in file_feature_dict.items():
        feats_tensor = torch.stack(data['features'])  # (num_segments, seq_len, hidden_size)
        labels_tensor = torch.stack(data['labels'])   # (num_segments, num_classes)

        torch.save({
            'features': feats_tensor,
            'labels': labels_tensor
        }, os.path.join(save_dir, f"{file_id}.pt"))


# swin = SwinWithLeadAttention(num_classes=num_classes).to(device)
# swin.to(device)
# criterion = nn.BCEWithLogitsLoss()
# swin_optimizer = optim.Adam(swin.parameters(), lr=1e-4, weight_decay=1e-4)
# # Learning rate scheduler
# swin_scheduler = optim.lr_scheduler.StepLR(swin_optimizer, step_size=5, gamma=0.5)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data's dir
mat_dir = "./Training_WFDB"
label_path = "./REFERENCE_multi_label.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 7
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
    checkpoint_dir = f'./KFold/Fold{fold}'
    checkpoint_dir2 = f'./KFold/Model/Fold{fold}'
    csv_file = f'./KFold/Fold{fold}/log.csv'
    checkpoint_path = os.path.join(checkpoint_dir2, f'swin_checkpoint_epoch{epoch}.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not exist: {checkpoint_path}")
        continue

    # Initializing model and load checkpoint
    model = SwinWithLeadAttention(num_classes=num_classes).to(device)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # load data
    train_loader, test_loader = get_dataloaders(mat_dir, label_path, train_files, test_files, csv_file)

    # extract feature and save
    extract_and_save_features(model, test_loader, device, checkpoint_dir)
    extract_and_save_features(model, train_loader, device, checkpoint_dir)


