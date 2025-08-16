import os
import json
import builtins
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import shutil

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load configuration JSON
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.json", help="putanja do JSON-a s hiper-parametrima")
args = parser.parse_args()

CFG = load_config(args.config)
BASE_RESULTS = CFG.get("results_dir", "results_exp_effnetv2")

# Set up results folder and logging
os.makedirs(BASE_RESULTS, exist_ok=True)
log_file = os.path.join(BASE_RESULTS, "dummy.txt")

def log_message(msg):
    lf = getattr(builtins, 'log_file', None)
    if lf is None:
        lf = log_file
    print(msg)
    with open(lf, "a") as f:
        f.write(msg + "\n")

# Seed and device setup
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.get("seed", 42))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learnable Neuromodulator module
class Neuromodulator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out = self.fc1(x.unsqueeze(0))
        out = self.relu(out)
        out = self.fc2(out)
        return torch.sigmoid(out).squeeze(0)

# Pairwise TinyImageNet dataset
class PairwiseImageNet(Dataset):
    def __init__(self, base_dataset, same_prob=0.5):
        super().__init__()
        self.base_dataset = base_dataset
        self.same_prob = same_prob
        # Use ImageFolder.samples to avoid loading all images during init (no I/O here)
        self.samples = self.base_dataset.samples  # list of (path, label)
        self.targets = [y for _, y in self.samples]

        self.class_dict = {}
        for idx, y in enumerate(self.targets):
            self.class_dict.setdefault(y, []).append(idx)

        self.labels_list = list(self.class_dict.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        imgA, labA = self.base_dataset[index]
        # If only one class exists, always pick the same class
        if len(self.labels_list) <= 1:
            labB = labA
        else:
            if random.random() < self.same_prob:
                labB = labA
            else:
                other_labels = [l for l in self.labels_list if l != labA]
                if not other_labels:
                    labB = labA
                else:
                    labB = random.choice(other_labels)
        idxB = random.choice(self.class_dict[labB])
        imgB, labB = self.base_dataset[idxB]
        return imgA, labA, imgB, labB

# Hebbian EfficientNetV2 model
class HebbianEffNetV2(nn.Module):
    """
    Wrap torchvision’s EfficientNet-V2-S for TinyImageNet (200 classes).
    We adapt stride of the first conv layer to 1 to avoid early downsampling
    and add an embedding head (fc1/fc2) similar to the CIFAR version.
    Returns logits, a list of conv activations (duplicated), and embedding.
    """
    def __init__(self, num_classes: int = 200, emb_dim: int = 512):
        super().__init__()
        base = models.efficientnet_v2_s(weights=None)
        # Adjust first conv stride to 1 for finer resolution on 64x64 inputs
        base.features[0][0].stride = (1, 1)
        # Replace classifier to output num_classes
        in_dim = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_dim, num_classes)
        )
        self.backbone = base
        self.conv1 = base.features[0][0]  # first conv for Hebbian reg
        # MLP head for embeddings (1280 input from EfficientNet)
        self.fc1 = nn.Linear(1280, emb_dim)
        self.bn_fc1 = nn.BatchNorm1d(emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.bn_fc2 = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_acts = []
        # Pass through backbone features
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i == 0:  # capture first conv activation
                conv_acts.append(x.clone())
        # Compute embedding from mean pooling prior to global pooling
        emb_input = torch.flatten(x.mean([2, 3]), 1)
        h = self.relu(self.bn_fc1(self.fc1(emb_input)))
        emb = self.relu(self.bn_fc2(self.fc2(h)))
        # Classifier head (logits)
        out = self.backbone.avgpool(x)
        out = torch.flatten(out, 1)
        logits = self.backbone.classifier(out)
        # Duplicate conv activation list to match conv_acts indexing in training
        if conv_acts:
            conv_acts = [conv_acts[0], conv_acts[0]]
        else:
            conv_acts = [x, x]
        return logits, conv_acts, emb

# Trainer class with best-model saving and full-metrics prints
class Trainer:
    def __init__(self, model, device, results_folder):
        self.model = model.to(device)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lambda_hebb = 0.1
        self.device = device
        self.results_folder = results_folder
        # Set number of classes and embedding dimension
        self.num_classes = model.backbone.classifier[-1].out_features
        self.embedding_dim = model.fc2.out_features
        # Setup accumulators for Hebbian update
        self.classwise_emb_sums = [torch.zeros(self.embedding_dim, device=device) for _ in range(self.num_classes)]
        self.classwise_counts = [0 for _ in range(self.num_classes)]
        self.learnable_modulator = Neuromodulator(input_dim=1, hidden_dim=8).to(device)
        self.old_params = None
        self.single_epoch_history = {"loss": [], "accuracy": []}
        self.pairwise_epoch_history = {"loss": [], "accuracy": []}
        self.best_val_acc = 0.0
        self.best_epoch = 0
        # Use conv1 as Hebbian conv layer
        self.conv_layer = model.conv1

    def save_model(self, filename="model.pth"):
        path = os.path.join(self.results_folder, filename)
        torch.save(self.model.state_dict(), path)
        log_message(f"Model saved to {path}")

    def neuromodulator(self, current_loss, target_loss=0.3):
        return self.learnable_modulator(current_loss - target_loss)

    def hebbian_reg_loss(self, conv_act):
        # Hebbian regularization on conv_layer
        act_mean = conv_act.mean(dim=0)
        loss = 0.0
        for f in range(self.conv_layer.out_channels):
            a_f = torch.nn.functional.adaptive_avg_pool2d(act_mean[f].unsqueeze(0), (3, 3)).squeeze(0).flatten()
            w_f = self.conv_layer.weight[f].mean(dim=0).flatten()
            loss += (a_f - w_f).pow(2).sum()
        return loss / self.conv_layer.out_channels

    def hebbian_update_fc2(self, fc2_layer):
        for c in range(self.num_classes):
            if self.classwise_counts[c] < 1:
                continue
            emb_mean = self.classwise_emb_sums[c] / self.classwise_counts[c]
            dW = 0.0001 * torch.ger(emb_mean, emb_mean)
            if dW.shape == fc2_layer.weight.data.shape:
                fc2_layer.weight.data += dW
        self.classwise_emb_sums = [torch.zeros(self.embedding_dim, device=self.device) for _ in range(self.num_classes)]
        self.classwise_counts = [0 for _ in range(self.num_classes)]

    def consolidation_loss(self, current_loss, alpha=1e-3):
        if self.old_params is None:
            return 0.0
        nm = self.neuromodulator(current_loss - 0.3)
        loss_c = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.old_params:
                loss_c += nn.functional.mse_loss(param, self.old_params[name])
        return alpha * nm * loss_c

    def evaluate_single_image(self, loader, cur_epoch=None, total_epochs=None):
        self.model.eval()
        running_loss, total, correct = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                logits, _, _ = self.model(data)
                loss = self.ce_loss(logits, labels)
                running_loss += loss.item() * data.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                total += data.size(0)
                correct += (preds == labels).sum().item()
        avg_loss = running_loss / total
        acc = 100.0 * correct / total
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        if cur_epoch and total_epochs:
            log_message(f"[P1-val] Epoch {cur_epoch}/{total_epochs}")
        log_message("=== Single-Image Evaluation ===")
        log_message(f"Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        log_message(f"Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
        log_message(f"Confusion Matrix:\n{cm}\n")
        self.model.train()
        return acc

    def train_single_image(self, train_loader, epochs=50, lr=1e-3, weight_decay=5e-4, test_loader=None, use_swa=False, patience=5):
        optimizer = optim.SGD(
            list(self.model.parameters()) + list(self.learnable_modulator.parameters()),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        if use_swa:
            swa_model = AveragedModel(self.model)
            swa_start = int(0.8 * epochs)
            swa_sched = SWALR(optimizer, swa_lr=lr, anneal_epochs=5, anneal_strategy="cos")
        best_val_acc = 0.0
        epochs_no_improv = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits, conv_acts, emb = self.model(data)
                loss = self.ce_loss(logits, labels)
                hebb = self.hebbian_reg_loss(conv_acts[1])
                total_loss = loss + self.lambda_hebb * self.neuromodulator(loss) * hebb
                total_loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)
                total += data.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                for i, lab in enumerate(labels):
                    self.classwise_emb_sums[lab] += emb[i].detach()
                    self.classwise_counts[lab] += 1
            self.hebbian_update_fc2(self.model.fc2)
            train_loss = running_loss / total
            train_acc = 100.0 * correct / total
            log_message(f"[P1] Epoch {epoch}/{epochs}  –  Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            if test_loader:
                val_acc = self.evaluate_single_image(test_loader, cur_epoch=epoch, total_epochs=epochs)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    epochs_no_improv = 0
                    self.save_model("best_phase1.pth")
                else:
                    epochs_no_improv += 1
                    if epochs_no_improv >= patience:
                        log_message(f">>> Early-stop @ epoch {epoch} (best {best_val_acc:.2f} %)")
                        break
            if use_swa and epoch > swa_start:
                swa_model.update_parameters(self.model)
                swa_sched.step()
            else:
                scheduler.step()
        if use_swa:
            update_bn(train_loader, swa_model, device=self.device)
            self.model = swa_model.module
        self.old_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}

    def evaluate_pairwise(self, loader, cur_epoch=None, total_epochs=None):
        self.model.eval()
        total, correct = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for a, la, b, lb in loader:
                a, la = a.to(self.device), la.to(self.device)
                b, lb = b.to(self.device), lb.to(self.device)
                oA, _, _ = self.model(a)
                oB, _, _ = self.model(b)
                pA, pB = oA.argmax(1), oB.argmax(1)
                correct += (pA == la).sum().item() + (pB == lb).sum().item()
                total += la.size(0) + lb.size(0)
                all_preds.extend(pA.cpu().tolist() + pB.cpu().tolist())
                all_labels.extend(la.cpu().tolist() + lb.cpu().tolist())
        acc = 100.0 * correct / total
        sk_acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        if cur_epoch and total_epochs:
            log_message(f"[P2-val] Epoch {cur_epoch}/{total_epochs}")
        log_message("=== Pairwise Evaluation ===")
        log_message(f"Batch Acc: {acc:.2f}%    Sklearn Acc: {sk_acc:.2f}")
        log_message(f"Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
        log_message(f"Confusion Matrix:\n{cm}\n")
        self.model.train()
        return sk_acc

    def train_pairwise(
            self,
            pairwise_loader,
            val_loader=None,
            epochs=60,
            lr=1e-4,
            weight_decay=1e-5,
            lambda_metric=0.15,
            margin=1.0,
            warmup_epochs=5,
            bn_recal_loader=None,      # <<< dodano: DataLoader s single-image batchovima za BN update
            freeze_stages=4            # <<< dodano: koliko ranih EfficientNetV2 feature blokova zamrznuti u warmupu
        ):
        # 1) (Privremeno) zamrzni rane blokove tijekom warmupa – stabilizira početak P2
        if freeze_stages > 0:
            for p in self.model.backbone.features[:freeze_stages].parameters():
                p.requires_grad = False

        # 2) Optimizator
        optimizer = optim.SGD(
            list(self.model.parameters()) + list(self.learnable_modulator.parameters()),
            lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay
        )

        # 3) LR warmup -> cosine
        after_warmup = max(1, epochs - warmup_epochs)
        linear = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=after_warmup)
        scheduler = SequentialLR(optimizer, [linear, cosine], milestones=[warmup_epochs])

        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            # 4) Rampa za lambdae (sprječava “metric domination” na startu)
            if epoch <= warmup_epochs:
                cur_lambda_metric = 0.0
                cur_lambda_hebb   = 0.0
            else:
                prog = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                cur_lambda_metric = lambda_metric * (1.0 - np.exp(-5.0 * prog))
                cur_lambda_hebb   = 0.03 * (1.0 - np.exp(-5.0 * prog))

            self.model.train()
            running, correct, total = 0.0, 0, 0

            for A, lA, B, lB in pairwise_loader:
                A, lA = A.to(self.device), lA.to(self.device)
                B, lB = B.to(self.device), lB.to(self.device)

                optimizer.zero_grad()

                oA, cA, eA = self.model(A)
                oB, cB, eB = self.model(B)

                # 5) Normaliziraj embeddinge za stabilne udaljenosti
                eA = F.normalize(eA, dim=1)
                eB = F.normalize(eB, dim=1)

                # 6) Gubici
                ce = self.ce_loss(oA, lA) + self.ce_loss(oB, lB)

                dists = (eA - eB).norm(dim=1)
                same  = (lA == lB)
                pos   = (dists[same] ** 2).sum() if same.any() else 0.0
                neg   = ((torch.clamp(margin - dists[~same], 0.0) ** 2).sum() if (~same).any() else 0.0)
                metric = (pos + neg) / A.size(0)

                hebb = (self.hebbian_reg_loss(cA[1]) + self.hebbian_reg_loss(cB[1])) / 2

                base = ce + cur_lambda_metric * metric
                loss = base + self.consolidation_loss(base) + cur_lambda_hebb * self.neuromodulator(base) * hebb

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)  # 7) grad clip
                optimizer.step()

                batch   = A.size(0)
                running += base.item() * batch
                correct += (oA.argmax(1) == lA).sum().item() + (oB.argmax(1) == lB).sum().item()
                total   += 2 * batch

            epoch_loss = running / total
            epoch_acc  = 100.0 * correct / total
            log_message(f"[P2] Epoch {epoch}/{epochs} – Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

            # 8) Nakon warmupa – odmrzni rane blokove (jednom)
            if epoch == warmup_epochs and freeze_stages > 0:
                for p in self.model.backbone.features[:freeze_stages].parameters():
                    p.requires_grad = True

            # 9) BN re-kalibracija na single-image loaderu (jako bitno za efnet)
            if bn_recal_loader is not None:
                update_bn(bn_recal_loader, self.model, device=self.device)

            scheduler.step()

            # 10) Validacija i spremanje najboljeg
            if val_loader is not None:
                val_sk = self.evaluate_pairwise(val_loader, cur_epoch=epoch, total_epochs=epochs)
                if val_sk > best_val_acc:
                    best_val_acc = val_sk
                    self.save_model("best_phase2.pth")

        log_message("Finished Phase 2.")

    def plot_training_history(self):
        if self.single_epoch_history["loss"]:
            plt.figure()
            plt.plot(self.single_epoch_history["loss"], label="Phase 1 Loss")
            plt.plot(self.single_epoch_history["accuracy"], label="Phase 1 Acc")
            plt.legend()
            plt.title("Phase 1")
            plt.xlabel("Epoch")
            plt.savefig(os.path.join(self.results_folder, "phase1.png"))
            plt.close()
        if self.pairwise_epoch_history["loss"]:
            plt.figure()
            plt.plot(self.pairwise_epoch_history["loss"], label="Phase 2 Loss")
            plt.plot(self.pairwise_epoch_history["accuracy"], label="Phase 2 Acc")
            plt.legend()
            plt.title("Phase 2")
            plt.xlabel("Epoch")
            plt.savefig(os.path.join(self.results_folder, "phase2.png"))
            plt.close()

    def save_metrics(self):
        metrics = {"phase1": self.single_epoch_history, "phase2": self.pairwise_epoch_history}
        with open(os.path.join(self.results_folder, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        log_message("Metrics saved.")

# t-SNE helper
def generate_tsne(cur_model: nn.Module, loader: DataLoader, fname: str, title: str = "t‑SNE"):
    cur_model.eval()
    all_embs, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            _, _, embs = cur_model(imgs)
            all_embs.append(embs.cpu())
            all_lbls.append(lbls)
    embs = torch.cat(all_embs).numpy()
    lbls = torch.cat(all_lbls).numpy()
    proj = TSNE(n_components=2, init="pca", random_state=42).fit_transform(embs)
    plt.figure(figsize=(8, 8))
    for c in np.unique(lbls):
        idx = lbls == c
        plt.scatter(proj[idx, 0], proj[idx, 1], s=5, alpha=0.6, label=str(c))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(BASE_RESULTS, fname)
    plt.savefig(out)
    plt.close()
    log_message(f"{title} saved to {out}")

# Main function for TinyImageNet training
def main():
    # Transforms for TinyImageNet (64x64) with ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        AutoAugment(AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # Dataset setup
    data_root = CFG["data_root"]
    train_ds = ImageFolder(os.path.join(data_root, "train"), transform=transform_train)
    # Prepare validation dataset: restructure val if necessary
    val_dir = os.path.join(data_root, "val")
    val_ann_file = os.path.join(val_dir, "val_annotations.txt")
    val_images_dir = os.path.join(val_dir, "images")
    fixed_val_dir = os.path.join(data_root, "val_fixed")
    if os.path.exists(val_ann_file) and os.path.exists(val_images_dir):
        if not os.path.exists(fixed_val_dir):
            os.makedirs(fixed_val_dir, exist_ok=True)
            with open(val_ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    label = parts[1]
                    class_dir = os.path.join(fixed_val_dir, label)
                    os.makedirs(class_dir, exist_ok=True)
                    src = os.path.join(val_images_dir, img_name)
                    dst = os.path.join(class_dir, img_name)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
        val_ds_path = fixed_val_dir
    else:
        val_ds_path = os.path.join(data_root, "val")
    val_ds = ImageFolder(val_ds_path, transform=transform_test)
    # Load TinyImageNet test set only if needed for t-SNE (no labels)
    test_ds_path = os.path.join(data_root, "test")
    if os.path.exists(test_ds_path):
        test_ds = ImageFolder(test_ds_path, transform=transform_test)
    # DataLoaders
    BATCH = CFG.get("batch_size", 128)
    train_loader_single = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader_single = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)
    train_loader_pw = DataLoader(PairwiseImageNet(train_ds, same_prob=0.5), batch_size=BATCH, shuffle=True, num_workers=4)
    if len(val_ds.classes) > 1:
        val_loader_pw = DataLoader(PairwiseImageNet(val_ds, same_prob=0.5), batch_size=BATCH, shuffle=False, num_workers=4)
    else:
        val_loader_pw = None
    # Training
    for run_id in range(1, CFG["num_runs"] + 1):
        run_dir = BASE_RESULTS if CFG["num_runs"] == 1 else os.path.join(BASE_RESULTS, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        # Each run writes its own log file
        builtins.log_file = os.path.join(run_dir, "training_log.txt")

        # Instantiate model and trainer
        model = HebbianEffNetV2(num_classes=200, emb_dim=512)
        trainer = Trainer(model, device, run_dir)

        # Path to the best Phase 1 checkpoint for this run
        best_phase1_path = os.path.join(run_dir, "best_phase1.pth")

        # If checkpoint exists, load it and skip Phase 1 training
        if os.path.exists(best_phase1_path):
            state_dict = torch.load(best_phase1_path, map_location=device)
            trainer.model.load_state_dict(state_dict)
            # Recompute consolidation baseline from loaded weights
            trainer.old_params = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
            log_message(f"Loaded Phase 1 checkpoint from {best_phase1_path}; skipping Phase 1 training.")
            trainer.best_epoch = -1  # indicator that Phase 1 was skipped
        else:
            # Otherwise perform Phase 1 training
            p1 = CFG["phase1"]
            trainer.train_single_image(
                train_loader_single,
                epochs=p1["epochs"],
                lr=p1["lr"],
                weight_decay=p1["weight_decay"],
                test_loader=val_loader_single,
                use_swa=p1["use_swa"],
                patience=p1["patience"],
            )
            log_message(
                f"Phase 1 završena na {trainer.best_epoch}. "
                f"Najbolja val-točnost: {trainer.best_val_acc:.2f}% (checkpoint učitan)."
            )
            # Visualize embeddings for Phase 1
            generate_tsne(model, val_loader_single, f"tsne_phase1_run{run_id}.png", title=f"t-SNE after Phase 1 (run {run_id})")

        # Always (re)load the best Phase 1 weights before Phase 2 to ensure correct starting point
        if os.path.exists(best_phase1_path):
            state_dict = torch.load(best_phase1_path, map_location=device)
            trainer.model.load_state_dict(state_dict)
            trainer.old_params = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

        # Phase 2 training
        p2 = CFG["phase2"]
        trainer.train_pairwise(
            pairwise_loader=train_loader_pw,
            val_loader=val_loader_pw,
            epochs=p2["epochs"],
            lr=p2["lr"],
            weight_decay=p2["weight_decay"],
            lambda_metric=p2["lambda_metric"],
            margin=p2["margin"],
            warmup_epochs=p2.get("warmup_epochs", 5),
            bn_recal_loader=train_loader_single,          # <<< NEW
            freeze_stages=p2.get("freeze_stages", 4),     # <<< NEW
        )
        log_message(
            ">>> Pairwise Evaluation on Held-out Validation Set <<<"
        )
        if val_loader_pw is not None:
            trainer.evaluate_pairwise(val_loader_pw)
        log_message(
            ">>> Single-Image Evaluation on Held-out Validation Set <<<"
        )
        trainer.evaluate_single_image(val_loader_single)
        trainer.plot_training_history()
        trainer.save_metrics()
        log_message(
            ">>> Generating t-SNE of validation embeddings <<<"
        )
        generate_tsne(model, val_loader_single, f"tsne_phase2_run{run_id}.png", title=f"t-SNE after Phase 2 (run {run_id})")

if __name__ == "__main__":
    main()