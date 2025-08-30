# (12, 3000) ---2D FFT(Fast Fourier transforms) spectrum 224 -> Pretrained model swintransform
# Spectrum is processed each image (12, F, T)
# #and resize (1, 256, 256) -> 1 chanel and get three times like # (3, 256, 256) that is input of model
# nperseg=128, noverlap=64 -> nperseg=512, noverlap=504
# loop x_batch[i]  -> output[] -> attended.mean-> out ()
# 12 images from 1 sample -> each image into model
# -> extract features -> concat → classify
# multi-class and multi-label
# (8,12,244,244) => (8*12,244,244)

import os
import math
import random
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, \
    recall_score
from torch.nn.functional import sigmoid
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from collections import Counter
import torchvision.models as models
# from torchvision.models import resnet50, ResNet50_Weights
from transformers import SwinForImageClassification, SwinModel, AutoImageProcessor
from tqdm import tqdm
import scipy
import cv2
from scipy.signal import spectrogram
from torch import amp
from torch.amp import GradScaler, autocast


class ECGDataset(Dataset):
    def __init__(self, mat_dir, label_path, file_list,
                 csv_file="./Swin_attention/log.csv", window_size=6 * 500,
                 img_size=224, is_test=False):
        self.mat_dir = mat_dir
        self.window_size = window_size
        self.img_size = img_size
        self.is_test = is_test
        self.csv_file = csv_file

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

            # print(mat_file, ":", signal_length, " ", label_group, " ", stride, " ", final_segments_shape)
            log_data.append([mat_file, signal_length, label_group, stride, final_segments_shape])

            for start_idx in start_indices:
                self.samples.append((mat_path, start_idx, label_group))

        df = pd.DataFrame(log_data,
                          columns=["mat_file", "signal_length", "label_group", "stride", "final_segments_shape"])
        df.to_csv(self.csv_file, index=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_path, start_idx, label = self.samples[idx]
        file_id = os.path.basename(mat_path).replace(".mat", "")
        mat_data = loadmat(mat_path)
        val_data = mat_data['val']

        segment = np.stack([val_data[lead_idx, start_idx:start_idx + self.window_size] for lead_idx in
                            range(val_data.shape[0])])  # (12, 3000)
        # Convert label_group to a multi-hot vector
        label_list = self.label_dict[file_id]  # Tuple of 3 labels
        label_vector = torch.zeros(num_classes)
        for lbl in label_list:
            if not pd.isna(lbl) and lbl != -1:
                label_vector[int(lbl) - 1] = 1

        # Spectrogram
        Sxx_all = []
        for lead in segment:
            f, t, Sxx = spectrogram(lead, fs=500, nperseg=512, noverlap=502)
            # Log transform
            Sxx_log = np.log1p(Sxx)
            # Nomalization
            Sxx_log = Sxx_log / (np.max(Sxx_log) + 1e-8)
            # Resize
            Sxx_resized = cv2.resize(Sxx_log, (self.img_size, self.img_size))
            Sxx_all.append(Sxx_resized)  # (12, F, T)

        Sxx_tensor = torch.tensor(np.array(Sxx_all), dtype=torch.float32)  # (12, 224, 224)
        # RGB 3 chanels
        Sxx_tensor = Sxx_tensor.unsqueeze(1)  # (12, 1, 224, 224)
        Sxx_tensor = Sxx_tensor.repeat(1, 3, 1, 1)  # (12, 3, 224, 224)

        return Sxx_tensor, label_vector.clone().detach().float(), file_id


# DataLoader function
def get_dataloaders(mat_dir, label_path, train_files, test_files, batch_size=10, img_size=224):
    train_dataset = ECGDataset(mat_dir, label_path, train_files, img_size=img_size)
    val_dataset = ECGDataset(mat_dir, label_path, val_files, img_size=img_size)
    test_dataset = ECGDataset(mat_dir, label_path, test_files, img_size=img_size, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers= 4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


class SwinWithLeadAttention(nn.Module):
    def __init__(self, num_classes=9, num_leads=12):
        super().__init__()
        self.num_leads = num_leads
        self.num_classes = num_classes

        # Load pretrained Swin model (not classifier)
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

    def forward(self, x):  # x: (B, 12, 3, 224, 224)
        B, L, C, H, W = x.shape
        assert L == self.num_leads, f"Expected {self.num_leads} leads, got {L}"

        # Merge batch and lead dimensions
        x = x.view(B * L, C, H, W)  # (B*12, 3, 224, 224)

        # Extract embeddings using Swin
        output = self.backbone(x)
        features = output.last_hidden_state  # (B*12, seq_len, hidden_size)

        # Debugging
        # print(f"Features shape after SwinModel: {features.shape}")  # (B*12, seq_len, hidden_size)

        # Split back into (B, L, seq_len, hidden_size)
        features = features.view(B, L, 49, self.hidden_size)  # (B, 12, 49, hidden_size)

        # Apply attention over leads
        attended = []
        for i in range(B):
            batch_leads = features[i]  # (L, seq_len, hidden_size)

            # Apply attention on the leads of this batch only
            attended_leads = self.lead_attention(batch_leads)  # (L, seq_len, hidden_size)

            attended.append(attended_leads)

        # Stack attended leads for all batches
        attended = torch.stack(attended, dim=0)  # (B, L, seq_len, hidden_size)

        # Fuse leads
        fused = attended.mean(dim=1)  # (B, seq_len, hidden_size)

        # Classify
        logits = self.classifier(fused.mean(dim=1))  # (B, num_classes)
        return logits

def train_model(train_loader, num_epochs=50):
    best_acc = {"swin": 0}
    # Initialize the GradScaler
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]:")

        for model, optimizer, scheduler, model_name in zip(
                [swin],
                [swin_optimizer],
                [swin_scheduler],
                ["Swin"]
        ):
            model.train()
            running_loss, correct, total = 0, 0, 0
            total_samples = 0
            epoch_loss = 0
            epoch_acc = 0
            epoch_acc_total = 0
            loop = tqdm(train_loader, desc=f"Training {model_name}")
            for x_batch, y_batch, _ in loop:
                x_batch, y_batch = x_batch.to(device), y_batch.to(
                    device)  # x_bacth -> (batch_size = 32, 12, 3, 244, 244)

                optimizer.zero_grad()
                with autocast("cuda"):
                    outputs = model(x_batch)  # (B, num_classes)
                    loss = criterion(outputs, y_batch)

                scaler.scale(loss).backward()  # Scale loss before backward
                scaler.step(optimizer)  # Update weights
                scaler.update()  # Update scaler

                running_loss += loss.item()
                total_samples += x_batch.size(0)

                preds = torch.sigmoid(outputs) > 0.5  # Multi-label prediction

                # Accuracy = trung binh so label dung moi sample
                correct_per_label = (preds == y_batch.to(preds.device)).float().mean().item()
                loop.set_postfix(loss=running_loss / total_samples, acc=round(correct_per_label, 4))
                epoch_loss = running_loss / total_samples
                epoch_acc_total += round(correct_per_label, 4)
            epoch_acc = epoch_acc_total / len(train_loader)
            val_loss, val_acc = evaluate_model(model, val_loader, model_name)
            print(f"Train - {model_name} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            print(f"Validation - {model_name} Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            scheduler.step()
            # torch.save(model.state_dict(), f"./Swin_attention/{epoch}_{model_name}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': swin.state_dict(),
                'optimizer_state_dict': swin_optimizer.state_dict(),
                'scheduler_state_dict': swin_scheduler.state_dict(),
                'loss': epoch_loss,
                'acc': epoch_acc,
            }, f"./Swin_attention/swin_checkpoint_epoch{epoch}.pth")

        torch.cuda.empty_cache()
    print("\nFinished")
def evaluate_model(model, val_loader, model_name):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    total_samples = 0
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)  # (B, num_classes)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            total_samples += x_batch.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
    all_preds = torch.cat(all_preds, dim=0)  # (total_samples, num_classes)
    all_targets = torch.cat(all_targets, dim=0)

    # Accuracy = trung binh so label dung moi sample
    correct_per_label = (all_preds == all_targets).float().mean().item()
    epoch_loss = running_loss / total_samples
    epoch_acc = round(correct_per_label, 4)

    return epoch_loss, epoch_acc


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
    df.to_csv(f"./Swin_attention/{model_name}_voting_results.csv", index=False)

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

    auc_mean = roc_auc_score(y_true, y_mean_probs, average="macro")
    recall_mean = recall_score(y_true, y_mean, average="macro", zero_division=0)
    accuracy_mean = (y_true == y_mean).astype(float).mean()
    f1_mean = f1_score(y_true, y_mean, average="macro", zero_division=0)

    auc_median = roc_auc_score(y_true, y_median_probs, average="macro")
    recall_median = recall_score(y_true, y_median, average="macro", zero_division=0)
    accuracy_median = (y_true == y_median).astype(float).mean()
    f1_median = f1_score(y_true, y_median, average="macro", zero_division=0)

    with open(f"./Swin_attention/{model_name}_voting_results.txt", "w") as f:
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


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data's dir
mat_dir = "./Training_WFDB"
label_path = "./REFERENCE_multi_label.csv"

# Get file .mat list
all_mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat")]
# Load label
df_labels = pd.read_csv(label_path)
label_dict = dict(zip(df_labels["recording"], df_labels["First_label"]))
# get list of labels corresponding to file .mat
all_labels = [label_dict[f.replace(".mat", "")] for f in all_mat_files if f.replace(".mat", "") in label_dict]
# Split train (80%), val (10%), test (10%)
train_files, test_files, train_labels, test_labels = train_test_split(
    all_mat_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
val_files, test_files, val_labels, test_labels = train_test_split(
    test_files, test_labels, test_size=0.5, random_state=42, stratify=test_labels)

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
# Load DataLoader
train_loader, val_loader, test_loader = get_dataloaders(mat_dir, label_path, train_files, val_files, test_files)


# Initializing model
num_classes = 9
swin = SwinWithLeadAttention(num_classes=num_classes).to(device)
swin.to(device)
criterion = nn.BCEWithLogitsLoss()
swin_optimizer = optim.Adam(swin.parameters(), lr=1e-4, weight_decay=1e-4)
swin_scheduler = optim.lr_scheduler.StepLR(swin_optimizer, step_size=5, gamma=0.5)
###############################///////////////////////
# # Restore model training from a previously saved checkpoint
# # Load checkpoint
# checkpoint = torch.load('swin_checkpoint_epoch7.pth')
#
# # Load state
# swin.load_state_dict(checkpoint['model_state_dict'])
# swin_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# swin_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#
# # get info epoch and loss
# start_epoch = checkpoint['epoch'] + 1
# previous_loss = checkpoint.get('loss', None)
#
# if previous_loss is not None:
#     print(f"Loaded from checkpoint at epoch {start_epoch - 1} with loss = {previous_loss:.4f}")
# else:
#     print(f"Loaded from checkpoint at epoch {start_epoch - 1}")
#####################################
# Train
train_model(train_loader, num_epochs=8)
# train_model(train_loader, num_epochs=start_epoch)
################
#Test
checkpoint_dir = './Swin_attention'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(checkpoint_dir, f"swin_checkpoint_epoch7.pth")

print(f"Test model from {checkpoint_path}")

# Initialization
model = SwinWithLeadAttention(num_classes=num_classes).to(device)
# Load state
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
evaluate_voting(model, test_loader, device, model_name=f"swin_epoch7")

#####################################
## Save loss and acc of train, test over each epoch
# def save_results_to_file(epoch, epoch_loss, epoch_acc, file_path="./Swin_attention/result_test_epoch.txt"):
#     with open(file_path, 'a') as f:
#         f.write(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}\n")
#
#
# def save_results_to_file2(epoch, epoch_loss, epoch_acc, file_path="./Swin_attention/result_train_epoch.txt"):
#     with open(file_path, 'a') as f:
#         f.write(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}\n")
# for epoch in range(0, 7):
#     checkpoint_path = os.path.join(checkpoint_dir, f"swin_checkpoint_epoch{epoch}.pth")
#
#     if not os.path.exists(checkpoint_path):
#         print(f"Checkpoint not exist: {checkpoint_path}")
#         continue
#
#     print(f"train model from {checkpoint_path}")# Result train model
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     save_results_to_file2(epoch, checkpoint['loss'],  checkpoint['acc'])

# for epoch in range(0, 7):
#     checkpoint_path = os.path.join(checkpoint_dir, f"swin_checkpoint_epoch{epoch}.pth")
#
#     if not os.path.exists(checkpoint_path):
#         print(f"Checkpoint not exist: {checkpoint_path}")
#         continue
#
#     print(f"test model from {checkpoint_path}") # Result test model
#
#     # Initialization
#     model = SwinWithLeadAttention(num_classes=num_classes).to(device)
#
#     # Load state
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     epoch_loss, epoch_acc = evaluate_model(model, test_loader, model_name=f"swin_epoch{epoch}")
#     save_results_to_file(epoch, epoch_loss, epoch_acc)
#     #evaluate_voting(model, test_loader, device, model_name=f"swin_epoch{epoch}")
# Thay doi fold = * và Fold*


