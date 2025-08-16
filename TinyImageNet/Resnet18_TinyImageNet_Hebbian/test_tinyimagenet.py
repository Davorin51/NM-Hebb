# test_tinyimagenet_resnet18.py
import os, argparse, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------
# Model: HebbianResNet18 (identičan tvom)
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

def _make_layer(in_planes, planes, num_blocks, stride):
    layers = [BasicBlock(in_planes, planes, stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(planes, planes, 1))
    return nn.Sequential(*layers)

class HebbianResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = _make_layer(64,  64, num_blocks=2, stride=1)
        self.layer2 = _make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = _make_layer(128,256, num_blocks=2, stride=2)
        self.layer4 = _make_layer(256,512, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 512); self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512); self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)
        self.conv2 = self.layer1[0].conv1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def forward(self, x):
        conv_acts = []
        x = self.conv1(x); conv_acts.append(x.clone())
        x = self.relu(self.bn1(x))
        res = x
        x = self.layer1[0].conv1(x); conv_acts.append(x.clone())
        x = self.relu(self.layer1[0].bn1(x))
        x = self.layer1[0].conv2(x); x = self.layer1[0].bn2(x)
        if self.layer1[0].downsample: res = self.layer1[0].downsample(res)
        x = self.relu(x + res)
        x = self.layer1[1](x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        emb = self.relu(self.bn_fc2(self.fc2(x)))
        logits = self.fc3(emb)
        return logits, conv_acts, emb

# -------------------------
# HF test dataset remap (robustno: 'wnid' ili 'label')
# -------------------------
class HFTinyImageNetTestRemap(Dataset):
    def __init__(self, hf_split, wnid_to_idx, transform=None):
        self.ds = hf_split
        self.transform = transform
        self.wnid_to_idx = wnid_to_idx
        colnames = self.ds.column_names
        feats = getattr(self.ds, "features", {})
        if "wnid" in colnames:
            def get_wnid(ex): return str(ex["wnid"])
        elif "label" in colnames:
            label_feat = feats.get("label", None)
            if hasattr(label_feat, "names") and label_feat.names:
                names = label_feat.names
                def get_wnid(ex):
                    v = ex["label"]
                    return str(names[v] if isinstance(v, int) else v)
            else:
                def get_wnid(ex): return str(ex["label"])
        else:
            raise ValueError("HF split nema ni 'wnid' ni 'label'.")
        self._get_wnid = get_wnid
        self.targets = []
        missing = set()
        for ex in self.ds:
            w = self._get_wnid(ex)
            if w not in self.wnid_to_idx:
                self.targets.append(-1); missing.add(w)
            else:
                self.targets.append(int(self.wnid_to_idx[w]))
        if missing:
            print(f"[WARN] {len(missing)} WNID-ova iz HF testa nije u train mappingu (npr {list(sorted(missing))[:10]}). "
                  f"Provjeri da --data_root pokazuje na isti TinyImageNet train kao za treniranje.")
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        y = self.targets[idx]
        if y < 0:
            raise KeyError("WNID iz HF testa nije pronađen u train mappingu.")
        return img, y

# -------------------------
# Helpers
# -------------------------
def build_train_mapping_from_imagenet_folder(train_root):
    tmp = ImageFolder(train_root)  # samo za class_to_idx
    # ImageFolder.class_to_idx je {wnid: idx} po abecedi — to je ono što je model koristio
    return tmp.class_to_idx

def flexible_load_state_dict(model, ckpt_path, device):
    # PyTorch 2.4+ može imati weights_only flag; fallback na stariji torch
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[INFO] Missing keys:", missing)
    if unexpected:
        print("[INFO] Unexpected keys:", unexpected)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="put do .pth checkpointa (Phase1 ili Phase2)")
    ap.add_argument("--data_root", required=True, help="korijen Tiny-ImageNet-a s train/ (isti kao u treningu)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # isti norm kao u treningu
    MEAN_TI = (0.480, 0.448, 0.398)
    STD_TI  = (0.277, 0.269, 0.282)
    test_tf = T.Compose([T.Resize(64), T.CenterCrop(64), T.ToTensor(), T.Normalize(MEAN_TI, STD_TI)])

    # 1) mapping iz train/
    train_root = os.path.join(args.data_root, "train")
    wnid_to_idx = build_train_mapping_from_imagenet_folder(train_root)

    # 2) HF test split
    hf = load_dataset("slegroux/tiny-imagenet-200-clean")
    test_ds = HFTinyImageNetTestRemap(hf["test"], wnid_to_idx, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # 3) model + ckpt
    model = HebbianResNet18(num_classes=200).to(device)
    flexible_load_state_dict(model, args.ckpt, device)
    model.eval()

    # 4) evaluacija
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            all_preds.append(logits.argmax(1).cpu())
            all_tgts.append(y.cpu())
    y_hat = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_tgts).numpy()

    acc  = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_hat, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_hat, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_hat)

    print(f"\n=== Tiny-ImageNet (HF test) — ResNet18 ===")
    print(f"Top-1 Acc: {acc*100:.2f}%")
    print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print(f"Confusion matrix shape: {cm.shape}")

if __name__ == "__main__":
    main()
