# test_tinyimagenet_densenet.py
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
#  Model: HebbianDenseNetTiny (identičan treningu)
# ----------------------------
class HebbianDenseNetTiny(nn.Module):
    def __init__(self, num_classes: int = 200, emb_dim: int = 512):
        super().__init__()
        self.base = models.densenet121(weights=None)

        # zamjena prvog conv-a: 3x3, stride=1; pool0 maknut
        new_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        self.base.features.conv0 = new_conv
        self.conv2 = self.base.features.conv0
        self.base.features.pool0 = nn.Identity()

        # makni classifier i napravi MLP head
        in_feat = self.base.classifier.in_features  # 1024
        self.base.classifier = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_feat, emb_dim)
        self.bn_fc1 = nn.BatchNorm1d(emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.bn_fc2 = nn.BatchNorm1d(emb_dim)
        self.fc3 = nn.Linear(emb_dim, num_classes)
        self.relu = nn.ReLU(inplace=True)

        self._conv2_act = None
        self.conv2.register_forward_hook(self._save_conv2_act)

    def _save_conv2_act(self, module, inp, out):
        self._conv2_act = out

    def forward(self, x):
        self._conv2_act = None
        x = self.base.features(x)
        conv_act = self._conv2_act if self._conv2_act is not None else x
        conv_acts = [conv_act, conv_act]
        x = self.avgpool(x).flatten(1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        emb = self.relu(self.bn_fc2(self.fc2(x)))
        logits = self.fc3(emb)
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
            def _get_wnid(ex): return str(ex["wnid"])
        elif "label" in cols:
            labfeat = feats.get("label", None)
            if hasattr(labfeat, "names") and labfeat.names:
                names = labfeat.names
                def _get_wnid(ex):
                    v = ex["label"]
                    return str(names[v] if isinstance(v, int) else v)
            else:
                def _get_wnid(ex): return str(ex["label"])
        else:
            raise ValueError("HF split nema ni 'wnid' ni 'label' kolonu.")
        self._get_wnid = _get_wnid

        self.targets = []
        missing = set()
        for ex in self.ds:
            w = self._get_wnid(ex)
            if w not in wnid_to_idx:
                self.targets.append(-1); missing.add(w)
            else:
                self.targets.append(int(wnid_to_idx[w]))
        if missing:
            print(f"[WARN] {len(missing)} WNID-ova nije nađeno u train mappingu, npr: {list(sorted(missing))[:10]}")

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        y = self.targets[idx]
        if y < 0:
            raise KeyError("WNID iz HF testa nije u train mappingu (provjeri --data_root).")
        return img, y

# ----------------------------
#  Helpers
# ----------------------------
def build_train_mapping(train_root):
    # identičan mapping kao u treningu (ImageFolder WNID -> idx)
    tmp = ImageFolder(train_root)
    return tmp.class_to_idx

def flexible_load_state_dict(model, ckpt_path, device):
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
    ap.add_argument("--ckpt", required=True, help="putanja do checkpointa (.pth) – Phase1/Phase2")
    ap.add_argument("--data_root", required=True, help="korijen originalnog Tiny-ImageNet foldera (s 'train/')")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # koristi iste statistike kao u treningu (ImageNet)
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    test_tf = T.Compose([
        T.Resize(64), T.CenterCrop(64),
        T.ToTensor(), T.Normalize(MEAN, STD)
    ])

    # 1) WNID -> idx mapping iz train/
    wnid_to_idx = build_train_mapping(os.path.join(args.data_root, "train"))

    # 2) HF dataset i test split
    hf = load_dataset("slegroux/tiny-imagenet-200-clean")
    test_ds = HFTinyImageNetTestRemap(hf["test"], wnid_to_idx, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # 3) Model + checkpoint
    model = HebbianDenseNetTiny(num_classes=200, emb_dim=512).to(device)
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

    print("\n=== Tiny-ImageNet (HF test) — DenseNet-121 (NM-Hebb) ===")
    print(f"Top-1 Acc: {acc*100:.2f}%")
    print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print(f"Confusion matrix shape: {cm.shape}")

if __name__ == "__main__":
    main()
