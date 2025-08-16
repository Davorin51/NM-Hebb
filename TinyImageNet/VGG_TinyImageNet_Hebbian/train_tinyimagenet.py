import os, json, random, argparse, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

###############################################################################
# Config loading and results setup
###############################################################################
def parse_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_tinyimagenet.json")
    return json.load(open(ap.parse_args().config))

CFG = parse_cfg()
PRINT_CM = False  # Global switch for confusion matrix logging
DATA_ROOT = CFG["data_root"]
BASE_RESULTS = "results_tinyimagenet"
os.makedirs(BASE_RESULTS, exist_ok=True)

log_file = os.path.join(BASE_RESULTS, "dummy.txt")  # placeholder

def log_message(msg):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

###############################################################################
# Utility: pretty logger
###############################################################################
def pretty_log(phase,           # "P1" ili "P2"
               split,           # "train" ili "val"
               epoch, n_epochs,
               loss, acc,
               prec=None, rec=None, f1=None,
               cm=None):
    tag = f"[{phase}]" if split == "train" else f"[{phase}-val]"
    if split == "train":
        msg = f"{tag} Epoch {epoch}/{n_epochs} – Loss: {loss:.4f}, Acc: {acc:.2f}%"
        log_message(msg)
    else:
        log_message(f"{tag} Epoch {epoch}/{n_epochs}")
        log_message("=== Single-Image Evaluation ===")
        log_message(f"Loss: {loss:.4f}, Acc: {acc:.2f}%")
        if prec is not None:
            log_message(f"Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
        if cm is not None and PRINT_CM:
            log_message("Confusion Matrix:")
            cm_rows = ["[" + " ".join(f"{v:4d}" for v in row) + "]" for row in cm]
            for r in cm_rows:
                log_message(r)

###############################################################################
# Set random seed and device
###############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Data Transforms and Constants
###############################################################################
MEAN_TI = (0.480, 0.448, 0.398)  # Tiny-ImageNet statistics
STD_TI = (0.277, 0.269, 0.282)

train_tf = T.Compose([
    T.RandomResizedCrop(64, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    AutoAugment(AutoAugmentPolicy.IMAGENET),
    T.ToTensor(),
    T.Normalize(MEAN_TI, STD_TI)
])

test_tf = T.Compose([
    T.Resize(64),
    T.CenterCrop(64),
    T.ToTensor(),
    T.Normalize(MEAN_TI, STD_TI)
])

from torchvision.datasets.folder import default_loader

###############################################################################
# Tiny-ImageNet Validation Dataset
###############################################################################
class TinyImageNetVal(ImageFolder):
    """Validation split has a single 'images/' folder + txt file with labels."""
    def __init__(self, root, transform=None):
        val_dir = os.path.join(root, "val")
        anno_file = os.path.join(val_dir, "val_annotations.txt")

        # 1) pročitaj mapiranje fajl->klasa
        fname2cls = {}
        with open(anno_file) as f:
            for ln in f:
                fname, cls, *_ = ln.split("\t")
                fname2cls[fname] = cls

        # 2) napravi privremenu folder-strukturu u memoriji
        classes = sorted(set(fname2cls.values()))
        class_to_idx = {c:i for i,c in enumerate(classes)}
        samples = [
            (os.path.join(val_dir, "images", fname), class_to_idx[cls])
            for fname, cls in fname2cls.items()
        ]

        self.loader = default_loader
        self.extensions = ('.jpeg', '.jpg', '.png')
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = self.imgs = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = None

###############################################################################
# Pairwise Dataset Wrapper
###############################################################################
class PairwiseDataset(Dataset):
    def __init__(self, base_dataset, same_prob=0.5):
        self.base_dataset = base_dataset
        self.same_prob = same_prob
        self.class_dict = {}
        for idx, (_, label) in enumerate(self.base_dataset):
            if label not in self.class_dict:
                self.class_dict[label] = []
            self.class_dict[label].append(idx)
        self.labels_list = list(self.class_dict.keys())
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, index):
        dataA, labelA = self.base_dataset[index]
        if random.random() < self.same_prob:
            labelB = labelA
            idxB = random.choice(self.class_dict[labelB])
        else:
            labelB = random.choice([l for l in self.labels_list if l != labelA])
            idxB = random.choice(self.class_dict[labelB])
        dataB, _ = self.base_dataset[idxB]
        return dataA, labelA, dataB, labelB

###############################################################################
# VGG‐style model with Hebbian blocks
###############################################################################
class HebbianConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=1, use_pool=True):
        super().__init__()
        layers, convs = [], []
        for _ in range(k):
            conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
            layers += [conv, nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            convs.append(conv)
            in_c = out_c
        if use_pool:
            layers.append(nn.MaxPool2d(2,2))
        self.block = nn.Sequential(*layers)
        self.convs = convs
    def forward(self, x):
        return self.block(x)

class HebbianVGG11(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.stage1 = HebbianConvBlock(3,64,k=1)
        self.stage2 = HebbianConvBlock(64,128,k=1)
        self.stage3 = HebbianConvBlock(128,256,k=2)
        self.stage4 = HebbianConvBlock(256,512,k=2)
        self.stage5 = HebbianConvBlock(512,512,k=2)
        self._convs = (
            self.stage1.convs +
            self.stage2.convs +
            self.stage3.convs +
            self.stage4.convs +
            self.stage5.convs
        )
        self.conv2 = self.stage3.convs[1]     # 256→256
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1     = nn.Linear(512,512)
        self.bn_fc1  = nn.BatchNorm1d(512)
        self.fc2     = nn.Linear(512,512)
        self.bn_fc2  = nn.BatchNorm1d(512)
        self.fc3     = nn.Linear(512,num_classes)
        self.relu    = nn.ReLU(inplace=True)
    def forward(self, x):
        conv_acts = []
        # ---------- stage-1 ----------
        x = self.stage1.convs[0](x)           # prvi conv (64→64)
        conv_acts.append(x.clone())           # conv_acts[0]
        x = self.stage1.block[1:](x)          # BN + ReLU + MaxPool

        # ---------- stage-2 ----------
        x = self.stage2(x)                    # cijeli stage-2 (64→128)

        # ---------- stage-3 ----------
        for m in self.stage3.block:           # prolazimo redom kroz module
            x = m(x)
            if m is self.conv2:               # kad dođemo do drugog conv-a
                conv_acts.append(x.clone())   # conv_acts[1]

        # ---------- ostatak ----------
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.avgpool(x).flatten(1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        emb = self.relu(self.bn_fc2(self.fc2(x)))
        logits = self.fc3(emb)

        return logits, conv_acts, emb

###############################################################################
# Trainer Class
###############################################################################
class Trainer:
    def __init__(self, model, device, results_folder):
        self.model = model.to(device)
        self.device = device
        self.results_folder = results_folder
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.target_loss = 0.3  # Fixed target loss for neuromodulation

    def get_neuromod_factor(self, current_loss):
        return float(torch.sigmoid(torch.tensor(self.target_loss - current_loss)))

    def save_model(self, filename="model.pth"):
        path = os.path.join(self.results_folder, filename)
        torch.save(self.model.state_dict(), path)
        log_message(f"Model saved to {path}")

    def hebbian_reg_loss_conv2(self, conv_act, conv_layer):
        """
        Hebb-regularizacija: kanal-po-kanal korelacija između srednjih
        aktivacija i (prostor-srednjih) težinskih vektora.
        """
        B, C, H, W = conv_act.shape
        act = conv_act.mean(dim=(2, 3))                 # B × C
        act = (act - act.mean(0)) / (act.std(0) + 1e-5)

        w = conv_layer.weight.mean(dim=(2, 3))          # K × C
        w = (w - w.mean(1, keepdim=True)) / (w.std(1, keepdim=True) + 1e-5)

        align = torch.mm(act, w.t()) / B                # B × K
        return align.abs().mean()

    def train_epoch(self, train_loader, optimizer, epoch, n_epochs=None):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            logits, conv_acts, _ = self.model(data)
            ce_loss = F.cross_entropy(logits, target)
            hebb_loss = self.hebbian_reg_loss_conv2(conv_acts[1], self.model.conv2)
            mod = self.get_neuromod_factor(ce_loss.item())
            batch_loss = ce_loss + mod * hebb_loss
            batch_loss.backward()
            optimizer.step()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_loss += batch_loss.item()
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        if n_epochs is not None:
            pretty_log("P1", "train", epoch, n_epochs, avg_loss, acc)
        return avg_loss, acc

    def validate(self, val_loader, epoch=None, n_epochs=None):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _, _ = self.model(data)
                val_loss += nn.CrossEntropyLoss()(output, target).item()
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        val_loss /= len(val_loader)
        acc = 100. * correct / total
        cm = confusion_matrix(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        if n_epochs is not None and epoch is not None:
            pretty_log("P1", "val", epoch, n_epochs, val_loss, acc, prec, rec, f1, cm)
        if acc > self.best_val_acc and epoch is not None:
            np.save(os.path.join(self.results_folder, f"cm_epoch{epoch}.npy"), cm)
        return val_loss, acc

    def train(self, train_loader, val_loader, epochs, lr, weight_decay, use_swa=False):
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        if use_swa:
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(optimizer, swa_lr=lr/5)
            swa_start = epochs//4
        best_val_acc = 0
        patience_counter = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch, n_epochs=epochs)
            val_loss, val_acc = self.validate(val_loader, epoch, n_epochs=epochs)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = val_acc
                self.save_model("best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
            else:
                scheduler.step()
            if patience_counter >= CFG["phase1"]["patience"]:
                log_message(f"Early stopping triggered after {epoch} epochs")
                break
        if use_swa:
            swa_model.to(self.device)
            update_bn(train_loader, swa_model, device=self.device)
            self.model.load_state_dict(swa_model.module.state_dict())
            self.save_model("swa_model.pth")

    def train_pairwise(self, train_loader, val_loader, epochs, lr, weight_decay, lambda_metric=0.5, margin=1.0):
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for A, lA, B, lB in train_loader:
                A, lA = A.to(self.device), lA.to(self.device)
                B, lB = B.to(self.device), lB.to(self.device)
                
                optimizer.zero_grad()
                
                outA, _, embA = self.model(A)
                outB, _, embB = self.model(B)
                
                # Compute distance-based loss
                d = F.pairwise_distance(embA, embB)
                same = (lA == lB).float()
                metric_loss = criterion(-d, same)  # Negativna udaljenost za BCE
                
                # Compute classification losses
                cls_loss = (F.cross_entropy(outA, lA) + F.cross_entropy(outB, lB)) / 2
                
                # Combined loss
                loss = lambda_metric * metric_loss + (1 - lambda_metric) * cls_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            # Validation
            val_loss, val_acc = self.validate_pairwise(val_loader)
            
            log_message(f'Epoch {epoch}: Train Loss: {total_loss/len(train_loader):.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = val_acc          # Update instance attribute
                self.save_model("best_model_phase2.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= CFG["phase2"]["patience"]:
                log_message(f"Early stopping triggered after {epoch} epochs")
                break

    def validate_pairwise(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for A, lA, B, lB in val_loader:
                A, lA = A.to(self.device), lA.to(self.device)
                B, lB = B.to(self.device), lB.to(self.device)
                
                outA, _, embA = self.model(A)
                outB, _, embB = self.model(B)
                
                d = F.pairwise_distance(embA, embB)
                same = (lA == lB).float()
                loss = criterion(-d, same)  # minus, isto kao u treningu
                
                total_loss += loss.item()
                pred = (d < 0.5).float()
                correct += (pred == same).sum().item()
                total += same.size(0)

        return total_loss / len(val_loader), 100. * correct / total


###############################################################################
# Evaluate checkpoint utility
###############################################################################
def evaluate_checkpoint(model_path, loader, device):
    mdl = HebbianVGG11(num_classes=200).to(device)
    mdl.load_state_dict(torch.load(model_path, map_location=device))
    mdl.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = mdl(x)
            all_preds.append(logits.argmax(1).cpu())
            all_tgts.append(y.cpu())
    y_hat = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_tgts).numpy()
    acc  = accuracy_score(y_true, y_hat)
    f1   = f1_score(y_true, y_hat, average='macro')
    prec = precision_score(y_true, y_hat, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_hat, average='macro', zero_division=0)
    return acc, prec, rec, f1

###############################################################################
# Helper for one run ----------------------------------------------------------
###############################################################################
def run_once(run_id: int,
             train_loader_single, val_loader_single, test_loader_single,
             train_loader_pw, val_loader_pw):
    run_folder = os.path.join(BASE_RESULTS, f"run_{run_id}")
    os.makedirs(run_folder, exist_ok=True)

    global log_file
    log_file = os.path.join(run_folder, "training.log")

    # 1) instanciraj model i trenera
    model   = HebbianVGG11(num_classes=200)
    trainer = Trainer(model, device, run_folder)

    # 2) PHASE 1
    log_message("\nPhase 1")
    trainer.train(train_loader_single,
                  val_loader_single,
                  epochs        = CFG["phase1"]["epochs"],
                  lr            = CFG["phase1"]["lr"],
                  weight_decay  = CFG["phase1"]["weight_decay"],
                  use_swa       = CFG["phase1"]["use_swa"])

    # 3) PHASE 2
    log_message("\nPhase 2")
    # Freeze backbone before Phase 2 (only FC layers trainable)
    for name, p in model.named_parameters():
        if name.startswith('stage1') or name.startswith('stage2'):
            p.requires_grad = False
    # Use quick-fix hyperparameters for phase 2
    trainer.train_pairwise(
        train_loader_pw,
        val_loader_pw,
        epochs        = CFG["phase2"].get("epochs", 30),
        lr            = CFG["phase2"].get("lr", 0.0005),
        weight_decay  = CFG["phase2"].get("weight_decay", 1e-4),
        lambda_metric = CFG["phase2"].get("lambda_metric", 0.15),
        margin        = CFG["phase2"].get("margin", 1.5)
    )

    # 4) evaluacija oba checkpointa
    acc1, p1, r1, f1_ = evaluate_checkpoint(
        os.path.join(run_folder, "best_model.pth"),        # Phase 1
        test_loader_single, device)
    acc2, p2, r2, f2_ = evaluate_checkpoint(
        os.path.join(run_folder, "best_model_phase2.pth"), # Phase 2
        test_loader_single, device)

    pretty_log("CMP1", "val", 0, 0, 0, acc1*100, p1, r1, f1_)
    pretty_log("CMP2", "val", 0, 0, 0, acc2*100, p2, r2, f2_)
    log_message(f"ΔAcc = {(acc2-acc1)*100:+.2f} pp | ΔF1 = {(f2_-f1_):+.3f}")

###############################################################################
# Main
###############################################################################
def main():
    # Create datasets
    train_set_full = ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
    val_set = TinyImageNetVal(DATA_ROOT, transform=test_tf)

    # Split training set into train and hold-out
    targets_np = np.array(train_set_full.targets)
    train_idx, hold_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=CFG["seed"])
        .split(np.zeros(len(targets_np)), targets_np)
    )
    train_set = Subset(train_set_full, train_idx)
    hold_set = Subset(train_set_full, hold_idx)

    # Create dataloaders
    train_loader_single = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_single = DataLoader(hold_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_single = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Pairwise dataloaders - manji batch za dupli pass
    train_loader_pw = DataLoader(PairwiseDataset(train_set, 0.5),
                               batch_size=48, shuffle=True, num_workers=4, pin_memory=True)  # Manji batch za GPU memoriju
    val_loader_pw = DataLoader(PairwiseDataset(hold_set, 0.5),
                             batch_size=48, shuffle=False, num_workers=4, pin_memory=True)

    # Pokreni jedan run (ili petlju za više runova)
    run_once(run_id=1,
             train_loader_single=train_loader_single,
             val_loader_single=val_loader_single,
             test_loader_single=test_loader_single,
             train_loader_pw=train_loader_pw,
             val_loader_pw=val_loader_pw)

if __name__ == "__main__":
    main()
