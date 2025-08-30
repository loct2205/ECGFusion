import os
import numpy as np
import torch
import pandas as pd
from config import config
from torch import nn, optim
from data_load import load_data
from models import duibi
from sklearn.metrics import roc_auc_score
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(filepath, model, optimizer=None):
    if os.path.isfile(filepath):
        print(f"Loading checkpoint from: {filepath}")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

def test_epoch(model, criterion, criterion_Cross, val_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    m_loss = 0
    outputs1 = []
    outputs2 = []
    outputs3 = []
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs1, target in val_dataloader:
            # inputs1 = inputs1 + torch.randn_like(inputs1) * 0.1
            inputs1 = inputs1.to(device)
            target = target.to(device)
            feat_ins, out_ins, feat_all, out = model(inputs1)
            feat_ins, out_ins, feat_all, out = feat_ins.to(torch.float), out_ins.to(torch.float), feat_all.to(
                torch.float), out.to(torch.float)
            loss1 = criterion(out_ins, target)
            loss2 = criterion(out, target)
            mutual_loss = criterion_Cross(feat_ins, feat_all) + criterion_Cross(feat_all, feat_ins)
            loss = loss1 * 0.75 + loss2 + 0.5 * mutual_loss
            loss_meter += loss.item()
            # m_loss += mutual_loss.item()/
            it_count += 1
            out1 = torch.sigmoid(out_ins)
            out = torch.sigmoid(out)
            for i in range(len(target)):
                outputs1.append(out1[i].cpu().detach().numpy())
                outputs.append(out[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())
        #     # scheduler_teacher_model.step()
        # auc1 = roc_auc_score(targets, outputs1)
        # TPR1 = utils.compute_TPR(targets, outputs1)
        acc1 = utils.compute_ACC(targets, outputs1)
        # f11 = utils.compute_F1(targets, outputs1)
        # auc = roc_auc_score(targets, outputs)
        # TPR = utils.compute_TPR(targets, outputs)
        acc = utils.compute_ACC(targets, outputs)
        # f1 = utils.compute_F1(targets, outputs)

        outputs1 = np.array(outputs1)
        outputs = np.array(outputs)
        preds1 = (outputs1 >= 0.5).astype(int)
        preds = (outputs >= 0.5).astype(int)

        # === For outputs1 ===
        auc1 = roc_auc_score(targets, outputs1, average='macro')
        TPR1 = recall_score(targets, preds1, average='macro')  # Macro TPR = macro recall
        #acc1 = accuracy_score(targets, preds1)
        f11 = f1_score(targets, preds1, average='macro')

        # === For outputs ===
        auc = roc_auc_score(targets, outputs, average='macro')
        TPR = recall_score(targets, preds, average='macro')  # Macro TPR = macro recall
        #acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')
    print('test_loss1: %.4f, m_loss1: %.4f, test_macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, m_loss / it_count, auc1, TPR1, acc1, f11))
    print('test_loss: %.4f, macro_auc: %.4f,  TPR: %.4f, acc: %.4f, f1: %.4f' % (
        loss_meter / it_count, auc, TPR, acc, f1))
    return auc1, TPR1, acc1, f11, auc, TPR, acc, f1

def test_only(fold, model_path, result_save_path):
    # Load data
    _, _, test_loader, num_classes = load_data(config.batch_size, fold=fold)

    # Init model
    model = duibi.Dui1(num_classes=num_classes).to(device)
    model = load_checkpoint(model_path, model)

    # Eval mode
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    criterion_Cross = nn.KLDivLoss(reduction="batchmean")  # or reuse DistillKL

    # Inference
    auc1, TPR1, acc1, f11, auc2, TPR2, acc2, f12 = test_epoch(model, criterion, criterion_Cross, test_loader)

    # Save to CSV
    result = {
        "fold": fold,
        "AUC_ins": auc1,
        "TPR_ins": TPR1,
        "ACC_ins": acc1,
        "F1_ins": f11,
        "AUC_all": auc2,
        "TPR_all": TPR2,
        "ACC_all": acc2,
        "F1_all": f12
    }

    df = pd.DataFrame([result])
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    if os.path.exists(result_save_path):
        df.to_csv(result_save_path, mode='a', index=False, header=False)
    else:
        df.to_csv(result_save_path, index=False)
    print(f"Saved test result for fold {fold} to {result_save_path}")
if __name__ == "__main__":
    result_csv = "/media/data3/users/loctx/MRM/results/result_summary.csv"
    for fold in range(5):
        model_path = f"/media/data3/users/loctx/MRM/Fold{fold}/Dui1_exp0_checkpoint_best.pth"
        test_only(fold=fold, model_path=model_path, result_save_path=result_csv)
