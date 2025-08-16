# test_tinyimagenet_effnetv2.py
import os, argparse, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision.models as models
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------
#  Model: HebbianEffNetV2 (isto kao u treningu)
# ----------------------------
class HebbianEffNetV2(nn.Module):
    def __init__(self, num_classes: int = 200, emb_dim: int = 512):
        super().__init__()
        base = models.efficientnet_v2_s(weights=None)
        # identične izmjene kao u treningu
        base.features[0][0].stride = (1, 1)
        in_dim = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_dim, num_classes)
        )
        self.backbone = base
        self.conv1 = base.features[0][0]
        self.fc1 = nn.Linear(1280, emb_dim)
        self.bn_fc1 = nn.BatchNorm1d(emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.bn_fc2 = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_acts = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i == 0:
                conv_acts.append(x.clone())
        emb_input = torch.flatten(x.mean([2, 3]), 1)
        h   = self.relu(self.bn_fc1(self.fc1(emb_input)))
        emb = self.relu(self.bn_fc2(self.fc2(h)))
        out = self.backbone.avgpool(x)
        out = torch.flatten(out, 1)
        logits = self.backbone.classifier(out)
        if conv_acts:
            conv_acts = [conv_acts[0], conv_acts[0]]
        else:
            conv_acts = [x, x]
        return logits, conv_acts, emb

# ----------------------------
#  HF Tiny-ImageNet test s remapom na train class_to_idx
# ----------------------------
class HFTinyImageNetTestRemap(Dataset):
    def __init__(self, hf_split, wnid_to_idx, transform=None):
        self.ds = hf_split
        self.transform = transform
        self.wnid_to_idx = wnid_to_idx

        cols = self.ds.column_names
        feats = getattr(self.ds, "features", {})
        if "wnid" in cols:
            def get_wnid(ex): return str(ex["wnid"])
        elif "label" in cols:
            labfeat = feats.get("label", None)
            if hasattr(labfeat, "names") and labfeat.names:
                names = labfeat.names
                def get_wnid(ex):
                    v = ex["label"]
                    return str(names[v] if isinstance(v, int) else v)
            else:
                def get_wnid(ex): return str(ex["label"])
        else:
            raise ValueError("HF split nema ni 'wnid' ni 'label' kolonu.")
        self._get_wnid = get_wnid

        self.targets = []
        missing = set()
        for ex in self.ds:
            w = self._get_wnid(ex)
            if w not in wnid_to_idx:
                self.targets.append(-1); missing.add(w)
            else:
                self.targets.append(int(wnid_to_idx[w]))
        if missing:
            print(f"[WARN] {len(missing)} WNID-ova nije nađeno u train mappingu. Primjeri: {list(sorted(missing))[:10]}")

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform: img = self.transform(img)
        y = self.targets[idx]
        if y < 0:
            raise KeyError("WNID iz HF testa nije u train mappingu (provjeri --data_root).")
        return img, y

# ----------------------------
#  Helpers
# ----------------------------
def build_train_mapping(train_root):
    # koristi isti mapping kao u treningu (ImageFolder class_to_idx: WNID -> idx)
    tmp = ImageFolder(train_root)
    return tmp.class_to_idx

def flexible_load_state_dict(model, ckpt_path, device):
    # izbjegni future warning, ali budi kompatibilan i sa starim PyTorch-om
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print("[INFO] Missing keys:", missing)
    if unexpected: print("[INFO] Unexpected keys:", unexpected)

# ----------------------------
#  Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="putanja do checkpointa (.pth) – Phase1 ili Phase2")
    ap.add_argument("--data_root", required=True, help="korijen originalnog Tiny-ImageNet foldera (s 'train/')")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VAŽNO: koristi iste norm. statistike kao u treningu (ImageNet)
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    test_tf = T.Compose([
        T.Resize(64), T.CenterCrop(64),
        T.ToTensor(), T.Normalize(MEAN, STD)
    ])

    # 1) WNID -> idx mapping iz train/
    wnid_to_idx = build_train_mapping(os.path.join(args.data_root, "train"))

    # 2) Učitaj HF dataset i izgradi test skup
    hf = load_dataset("slegroux/tiny-imagenet-200-clean")
    test_ds = HFTinyImageNetTestRemap(hf["test"], wnid_to_idx, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # 3) Model + checkpoint
    model = HebbianEffNetV2(num_classes=200, emb_dim=512).to(device)
    flexible_load_state_dict(model, args.ckpt, device)
    model.eval()

    # 4) Evaluacija
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

    print("\n=== Tiny-ImageNet (HF test) — EfficientNetV2-S (NM-Hebb) ===")
    print(f"Top-1 Acc: {acc*100:.2f}%")
    print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print(f"Confusion matrix shape: {cm.shape}")

if __name__ == "__main__":
    main()
