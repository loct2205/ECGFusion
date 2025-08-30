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
import torchvision.models as models
from tqdm import tqdm


class ECGFusionDataset(Dataset):
    def __init__(self, file_ids, label_path, resnet_dir, swin_dir, csv_file="./KFold/AttentionModel/Fold3/log.csv",
                 num_classes=9, fixed_segment_len=18):
        self.file_ids = file_ids
        self.resnet_dir = resnet_dir
        self.swin_dir = swin_dir
        self.num_classes = num_classes
        self.csv_file = csv_file
        self.fixed_segment_len = fixed_segment_len

        # Load label
        df_labels = pd.read_csv(label_path)
        self.label_dict = dict()
        for _, row in df_labels.iterrows():
            labels = [row["First_label"], row["Second_label"], row["Third_label"]]
            self.label_dict[row["recording"]] = tuple([lbl if not np.isnan(lbl) else -1 for lbl in labels])

        log_data = [[fid] for fid in file_ids]
        df = pd.DataFrame(log_data, columns=['file_ids'])
        df.to_csv(self.csv_file, index=False)

    def __len__(self):
        return len(self.file_ids)

    def mean_pooling(self, feat, target_len):
        seg_len = feat.shape[0]
        if seg_len == target_len:
            return feat

        # segment > target_len, target_len groups, mean each group
        if seg_len > target_len:
            # split segment to target_len parts, mean each part
            bins = torch.linspace(0, seg_len, steps=target_len + 1).long()
            pooled = []
            for i in range(target_len):
                start, end = bins[i].item(), bins[i+1].item()
                pooled.append(feat[start:end].mean(dim=0))
            return torch.stack(pooled, dim=0)

        # segment < target_len, cant pooling, can repeat or padding
        #  repeat to avoid padding zero
        repeat_factor = target_len // seg_len
        remainder = target_len % seg_len
        feat_repeated = feat.repeat(repeat_factor, *([1] * (feat.dim() - 1)))
        if remainder > 0:
            feat_repeated = torch.cat([feat_repeated, feat[:remainder]], dim=0)
        return feat_repeated

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label_list = self.label_dict[file_id]  # Tuple of 3 labels
        label_vector = torch.zeros(self.num_classes)
        for lbl in label_list:
            if not pd.isna(lbl) and lbl != -1:
                label_vector[int(lbl) - 1] = 1

        # Load ResNet feature
        resnet_path = os.path.join(self.resnet_dir, f"{file_id}.npy")
        resnet_data = np.load(resnet_path, allow_pickle=True).item()
        resnet_feat = torch.tensor(resnet_data["features"], dtype=torch.float32)  # (segment, 512)

        # Load Swin feature
        swin_path = os.path.join(self.swin_dir, f"{file_id}.pt")
        swin_data = torch.load(swin_path)
        swin_feat = swin_data["features"].float()  # (segment, seq_len, 768)


        resnet_feat = self.mean_pooling(resnet_feat, self.fixed_segment_len)  # (fixed_segment_len, 512)
        swin_feat = self.mean_pooling(swin_feat, self.fixed_segment_len)      # (fixed_segment_len, seq_len, 768)

        label = torch.tensor(label_vector, dtype=torch.float32)

        return resnet_feat, swin_feat, label, file_id



def get_dataloaders(train_files, test_files, label_path,
                    resnet_dir="/media/data3/users/minhnb/Fold3/feature", swin_dir="./KFold/Feature/Fold3",
                    batch_size=64, num_workers=4):
    train_dataset = ECGFusionDataset(train_files, label_path, resnet_dir, swin_dir, csv_file="./KFold/AttentionModel/Fold3/log_train.csv")
    test_dataset = ECGFusionDataset(test_files, label_path, resnet_dir, swin_dir, csv_file="./KFold/AttentionModel/Fold3/log_test.csv")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def CT_Multi(feature, depth, output, weights=None, w=0.2, t=0.07, e=0.2, coef=1):
    def compute_comparison_matrix(depth):
        # Initialize the comparison matrix
        comparison_matrix = torch.zeros((depth.shape[0], depth.shape[0]), dtype=torch.float32)
        num_classes = depth.shape[1]
        # Compare each pair of labels
        for i in range(depth.shape[0]):
            for j in range(depth.shape[0]):
                comparison_matrix[i, j] = torch.sum(depth[i] == depth[j]) / num_classes
        return comparison_matrix

    k = feature.reshape([feature.shape[0], -1])  # (batch_size, feature_dim)
    q = feature.reshape([feature.shape[0], -1])  # (batch_size, feature_dim)

    depth = depth.reshape(depth.shape[0], -1)  # target (batch_size, target_dim)

    l_dist = compute_comparison_matrix(depth).to(depth.device)

    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)

    # dot product of anchor with positives. Positives are keys with similar label
    threshold = coef / depth.shape[1]
    pos_i = ((l_dist.ge(threshold)))
    neg_i = (~(l_dist.ge(threshold)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0

    prod = torch.einsum("nc,kc->nk", [q, k]) / t  # (batch_size, batch_size): dot product of query and key

    pos = prod * pos_i
    neg = prod * neg_i
    #  Pushing weight

    if weights is not None:
        pushing_w = weights.to(depth.device)

        # Sum exp of negative dot products
        neg_exp_dot = (pushing_w * (torch.exp(neg)) * (neg_i)).sum(1)
    else:
        neg_exp_dot = (torch.exp(neg) * (neg_i)).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom = pos_i.sum(1)

    # Avoid division by zero
    denom[denom == 0] = 1

    loss = ((-torch.log(torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (pos_i)).sum(
        1) / denom)
    loss = ((loss * no_neg_flag).unsqueeze(-1)).mean()

    return loss

class FeatureProjector(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in)
        self.activation = nn.GELU()
        #self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

class ecg_fusion_model(nn.Module):
    def __init__(self, resnet_dim=512, swin_dim=768, hidden_dim=256, num_heads=8, num_classes=9):
        super().__init__()
        # projector to dimension
        self.resnet_proj = FeatureProjector(resnet_dim, hidden_dim)
        self.swin_proj = FeatureProjector(swin_dim, hidden_dim)

        # Cross attention
        self.cross_attn_R_to_S = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_S_to_R = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Fusion feature and classifer
        self.fusion_proj = nn.Linear(hidden_dim * 2, 1024)  #out_dim=1024
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, resnet_feat, swin_feat):
        """
        resnet_feat: (B, segment, 512)
        swin_feat: (B, segment, seq_len, 768)
        """
        #print(swin_feat.shape)
        B, segment, seq_len, swin_dim = swin_feat.shape
        # add batch dimension
        R = resnet_feat  # (B, Lr, 512)
        S = swin_feat.reshape(B, segment * seq_len, -1)  # (B, Ls, 768)

        # Projector same dim
        R_proj = self.resnet_proj(R)  # (B, segment, 512)
        S_proj = self.swin_proj(S)  # (B, segment * seq_len, 768)

        # Cross attention
        R_to_S, _ = self.cross_attn_R_to_S(query=R_proj, key=S_proj, value=S_proj) #query=R_proj, key=R_proj, value=S_proj
        S_to_R, _ = self.cross_attn_S_to_R(query=S_proj, key=R_proj, value=R_proj) #query=S_proj, key=S_proj, value=R_proj
        # R_to_S, _ = self.cross_attn_R_to_S(query=R_proj, key=R_proj, value=S_proj)
        # S_to_R, _ = self.cross_attn_S_to_R(query=S_proj, key=S_proj, value=R_proj)

        # Mean pooling and fusion
        pooled_R = R_to_S.mean(dim=1)  # (B, 256)
        pooled_S = S_to_R.mean(dim=1)  # (B, 256)

        fusion = torch.cat([pooled_R, pooled_S], dim=-1)  # (B, 512)
        fusion = self.fusion_proj(fusion)  # (B, 1024)

        logits = self.classifier(fusion)  # (B, num_classes)
        return logits, fusion


def train_fusion_model(model, train_loader, val_loader, device, num_epochs=30, lr=1e-4,
                       weight_decay=1e-4, early_stop_patience=5, save_path="./KFold/AttentionModel/Fold3/best_fusion_model.pth",
                       log_txt_path="./KFold/AttentionModel/Fold3/fusion_training_log.txt",
                       ct_loss_weight=0.2):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0
    best_epoch = -1
    patience = 0

    with open(log_txt_path, "w") as log_file:
        log_file.write("Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\tVal F1\n")

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_losses = []
            train_accs = []

            for resnet_feat, swin_feat, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
                resnet_feat = resnet_feat.to(device)
                swin_feat = swin_feat.to(device)
                labels = labels.to(device).float()

                batch_size = resnet_feat.size(0)
                logits, fusion = model(resnet_feat, swin_feat)

                bce_loss = criterion(logits, labels)
                ct_loss = CT_Multi(fusion, labels, logits)
                loss = bce_loss + ct_loss_weight * ct_loss
                probs = torch.sigmoid(logits).detach()
                preds = (probs > 0.5).float()
                acc = (preds == labels).float().mean().item()
                train_accs.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_acc = np.mean(train_accs)
            avg_train_loss = np.mean(train_losses)

            # Validation
            model.eval()
            val_accs, val_losses, preds_all, labels_all = [], [], [], []
            with torch.no_grad():
                for resnet_feat, swin_feat, labels, _ in val_loader:
                    resnet_feat = resnet_feat.to(device)
                    swin_feat = swin_feat.to(device)
                    labels = labels.to(device).float()

                    batch_size = resnet_feat.size(0)

                    logits, _ = model(resnet_feat, swin_feat)

                    loss = criterion(logits, labels)
                    val_losses.append(loss.item())

                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(np.float32)
                    acc = (preds == labels.cpu().numpy()).mean()
                    val_accs.append(acc)
                    preds_all.extend(preds)
                    labels_all.extend(labels.cpu().numpy())

            avg_val_acc = np.mean(val_accs)
            avg_val_loss = np.mean(val_losses)
            val_f1 = f1_score(np.array(labels_all), np.array(preds_all), average='macro', zero_division=0)

            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Train Acc = {avg_train_acc:.4f} | "
                  f"Val Loss = {avg_val_loss:.4f} | Val Acc = {avg_val_acc:.4f} | Val F1 = {val_f1:.4f}")

            log_file.write(f"{epoch}\t{avg_train_loss:.4f}\t{avg_train_acc:.4f}\t"
                           f"{avg_val_loss:.4f}\t{avg_val_acc:.4f}\t{val_f1:.4f}\n")

            # Early Stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved at epoch {epoch}")
            else:
                patience += 1
                # if patience >= early_stop_patience:
                #     print(f"Early stopping at epoch {epoch}")
                #     break

        # best_model
        log_file.write(f"\nBest model at epoch {best_epoch} with Val F1 = {best_val_f1:.4f}\n")
        print(f"Training completed. Best F1: {best_val_f1:.4f} at epoch {best_epoch}")


def evaluate_voting(model, test_loader, device, model_name, threshold=0.5):
    model.to(device)
    model.eval()
    frame_predictions = {}
    frame_probabilities = {}

    with torch.no_grad():
        for resnet_feat, swin_feat, labels, file_id_batch in test_loader:
            resnet_feat = resnet_feat.to(device)
            swin_feat = swin_feat.to(device)
            labels = labels.to(device).float()

            batch_size = resnet_feat.size(0)
            logits_batch = []

            logits, _ = model(resnet_feat, swin_feat)  # (1, num_classes)
            logits_batch.append(logits)

            outputs = torch.cat(logits_batch, dim=0)  # (B, num_classes)
            probs = torch.sigmoid(outputs)  # (batch_size, num_classes)
            #probs = logits.detach()
            preds = (probs > 0.5).float()

            for i in range(batch_size):
                file_id = file_id_batch[i]  # ID of file ECG
                if file_id not in frame_predictions:
                    frame_predictions[file_id] = []
                    frame_probabilities[file_id] = []

                frame_predictions[file_id].append(preds[i].cpu().numpy())
                frame_probabilities[file_id].append(probs[i].cpu().numpy())

    majority_voting, mean_voting, median_voting, mean_probs_dict, median_probs_dict = {}, {}, {}, {}, {}

    for file_id in frame_predictions:
        print(file_id)
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
    df.to_csv(f"./KFold/AttentionModel/Fold3/{model_name}_voting_results.csv", index=False)

    # Compute evaluation metrics
    y_true_dict = {}
    for _, _, y_batch, file_id_batch in test_loader:
        for i in range(len(y_batch)):
            file_id = file_id_batch[i]
            if file_id not in y_true_dict:
                y_true_dict[file_id] = y_batch[i].cpu().numpy()
    print('continue')
    y_true = np.array(list(y_true_dict.values()))
    y_mean = np.array(list(mean_voting.values()))
    y_median = np.array(list(median_voting.values()))

    # Compute performance metrics
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

    with open(f"./KFold/AttentionModel/Fold3/{model_name}_voting_results.txt", "w") as f:
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
    # match labels by order (true vs predicted)
    y_true_mean_o, y_pred_mean_o = match_true_pred_by_order(y_true, y_mean_probs, threshold=0.5)
    y_true_median_o, y_pred_median_o = match_true_pred_by_order(y_true, y_median_probs, threshold=0.5)
    plot_confusion_matrix(y_true_single, y_mean_single,
                          f"{model_name} - Mean Voting - Confusion Matrix",
                          f"./KFold/AttentionModel/Fold3/{model_name}_mean_confusion_matrix.png")

    plot_confusion_matrix(y_true_single, y_median_single,
                          f"{model_name} - Median Voting - Confusion Matrix",
                          f"./KFold/AttentionModel/Fold3/{model_name}_median_confusion_matrix.png")
    plot_confusion_matrix(
        y_true_mean_o, y_pred_mean_o,
        title=f"{model_name} - Mean Voting - Confusion Matrix",
        filename=f"./KFold/AttentionModel/Fold3/confusion_matrix/{model_name}_mean_confusion_matrix.png"
    )

    plot_confusion_matrix(
        y_true_median_o, y_pred_median_o,
        title=f"{model_name} - Median Voting - Confusion Matrix",
        filename=f"./KFold/AttentionModel/Fold3/confusion_matrix/{model_name}_median_confusion_matrix.png"
    )
    print("Completed!")
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

def match_true_pred_by_order(y_true, y_probs, threshold=0.5):

    y_true_labels = []
    y_pred_labels = []

    for i in range(len(y_true)):
        true_indices = np.where(y_true[i] == 1)[0]
        pred_indices = np.where(y_probs[i] >= threshold)[0]

        match_len = min(len(true_indices), len(pred_indices))
        for j in range(match_len):
            y_true_labels.append(true_indices[j])
            y_pred_labels.append(pred_indices[j])

    return np.array(y_true_labels), np.array(y_pred_labels)

# Data's dir
mat_dir = "./Training_WFDB"
label_path = "./REFERENCE_multi_label.csv"

df = pd.read_csv('./KFold/REFERENCE_5fold.csv')
fold = 3
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

# Load DataLoader
train_loader, test_loader = get_dataloaders(train_files, test_files, label_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ecg_fusion_model(
    resnet_dim=512,
    swin_dim=768,
    hidden_dim=256,
    num_heads=8,
    num_classes=9
)

train_fusion_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    device=device,
    num_epochs=15,
    lr=1e-4,
    weight_decay=1e-4,
    early_stop_patience=5,
)
state_dict = torch.load("./KFold/AttentionModel/Fold3/best_fusion_model.pth", map_location=device)
model.load_state_dict(state_dict)
evaluate_voting(model, test_loader, device, model_name = "best_fusion_model_fold3")





