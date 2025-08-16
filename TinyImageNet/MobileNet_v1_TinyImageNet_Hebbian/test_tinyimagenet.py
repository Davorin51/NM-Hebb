import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --------- Model (isti layout kao u trening skripti) ---------
class HebbianMobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 200, emb_dim: int = 512):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        # first block
        self.conv1 = backbone.features[0][0]
        self.bn1   = backbone.features[0][1]
        self.relu1 = nn.ReLU6(inplace=True)
        # dodatni conv za Hebbian reg (kao u tvom kodu)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU6(inplace=True)
        # ostatak feat
        self.features = backbone.features[1:]
        # head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1280, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.fc3 = nn.Linear(emb_dim, num_classes)
        # za kompatibilnost s treniranjem
        self.saved_acts = []
        self.conv2.register_forward_hook(self._save_conv2_act)

    def _save_conv2_act(self, module, inp, out):
        self.saved_acts.append(out)

    def forward(self, x):
        self.saved_acts = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        emb = self.fc2(x)
        logits = self.fc3(emb)
        conv_act = self.saved_acts[0] if self.saved_acts else emb
        conv_acts = [conv_act, conv_act]
        return logits, conv_acts, emb

# --- zamjena za HFTinyImageNetTestRemap ---
class HFTinyImageNetTestRemap(Dataset):
    """
    HF tiny-imagenet-200-clean 'test' -> torch Dataset s labelama.
    Robusno detektira je li split ima 'wnid' ili 'label' te vraća WNID string,
    pa ga mapira na ImageFolder(train) index.
    """
    def __init__(self, hf_split, wnid_to_idx, transform=None):
        self.ds = hf_split
        self.transform = transform
        self.wnid_to_idx = wnid_to_idx

        colnames = self.ds.column_names
        feats = getattr(self.ds, "features", {})

        # detekcija izvora WNID-a
        if "wnid" in colnames:
            def get_wnid(ex): 
                return str(ex["wnid"])
        elif "label" in colnames:
            # ako je ClassLabel s imenima (WNID stringovi)
            label_feat = feats.get("label", None)
            if hasattr(label_feat, "names") and label_feat.names:
                names = label_feat.names  # lista WNID stringova
                def get_wnid(ex):
                    v = ex["label"]
                    if isinstance(v, int):
                        return str(names[v])
                    # ako je već string
                    return str(v)
            else:
                # 'label' je već string WNID
                def get_wnid(ex):
                    return str(ex["label"])
        else:
            raise ValueError("HF split nema ni 'wnid' ni 'label' kolonu.")

        self._get_wnid = get_wnid

        # pripremi targets + sanity-check na mapping
        self.targets = []
        missing = set()
        for ex in self.ds:
            w = self._get_wnid(ex)
            if w not in self.wnid_to_idx:
                missing.add(w)
                # stavi privremeno -1 da se kasnije lakše debagira
                self.targets.append(-1)
            else:
                self.targets.append(int(self.wnid_to_idx[w]))

        if missing:
            msg = (
                f"Upozorenje: {len(missing)} WNID-ova iz HF testa nije u train mappingu.\n"
                f"Npr: {list(sorted(missing))[:10]}\n"
                f"Provjeri da '--data_root' pokazuje na ISTI TinyImageNet train koji si koristio za treniranje."
            )
            print(msg)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        y = self.targets[idx]
        if y < 0:
            # ako je došlo do mismatcha, digni error s korisnom porukom
            raise KeyError("WNID iz HF testa nije pronađen u train mappingu. "
                           "Vidi upozorenje ispisano pri inicijalizaciji.")
        return img, y

# --------- evaluacija ---------
@torch.no_grad()
def evaluate_single(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        total += x.size(0)

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)
    print(f"[TEST] Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | P: {prec:.2f} R: {rec:.2f} F1: {f1:.2f}")
    print("Confusion matrix shape:", cm.shape)
    return acc

def build_train_mapping_from_imagenet_folder(train_root):
    """
    Vrati wnid->idx mapping identičan onome koji je ImageFolder koristio u treningu.
    """
    imgf = ImageFolder(train_root)
    # ImageFolder.class_to_idx: dict(wnid -> idx)
    return imgf.class_to_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="putanja do .pth checkpointa (best_phase1.pth ili best_phase2.pth)")
    ap.add_argument("--data_root", required=True, help="putanja do originalnog Tiny-ImageNet root-a (radi train mappinga)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emb_dim", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) Transforms – isti kao validacijski u treningu
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 2) Učitaj train mapping (wnid -> index) iz tvog train foldera
    train_root = os.path.join(args.data_root, "train")
    wnid_to_idx = build_train_mapping_from_imagenet_folder(train_root)
    
    # 3) Učitaj HF dataset i napravi test DataLoader
    hf = load_dataset("slegroux/tiny-imagenet-200-clean")
    test_ds = HFTinyImageNetTestRemap(hf["test"], wnid_to_idx, transform=transform_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    # 4) Model i checkpoint
    model = HebbianMobileNetV1(num_classes=200, emb_dim=args.emb_dim).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")

    # 5) Evaluacija
    evaluate_single(model, test_loader, device)

if __name__ == "__main__":
    main()
