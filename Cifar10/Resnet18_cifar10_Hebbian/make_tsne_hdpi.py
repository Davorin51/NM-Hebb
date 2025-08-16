# make_tsne_hdpi.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# ---------------------------
#  CIFAR-ResNet18 (isti raspored kao u treningu)
# ---------------------------
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
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = _make_layer(64,  64, 2, 1)
        self.layer2 = _make_layer(64, 128, 2, 2)
        self.layer3 = _make_layer(128,256, 2, 2)
        self.layer4 = _make_layer(256,512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1    = nn.Linear(512, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2    = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3    = nn.Linear(512, num_classes)
        self.conv2  = self.layer1[0].conv1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        res = x
        x = self.layer1[0].conv1(x)
        x = self.layer1[0].bn1(x); x = self.layer1[0].relu(x)
        x = self.layer1[0].conv2(x); x = self.layer1[0].bn2(x)
        if self.layer1[0].downsample: res = self.layer1[0].downsample(res)
        x = self.layer1[0].relu(x + res)
        x = self.layer1[1](x)
        x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        emb = self.relu(self.bn_fc2(self.fc2(x)))
        logits = self.fc3(emb)
        return logits, None, emb

# ---------------------------
#  Embedding extraction + t-SNE
# ---------------------------
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for x, y in loader:
        x = x.to(device)
        _, _, e = model(x)
        embs.append(e.cpu()); lbls.append(y.clone())
    return torch.cat(embs).numpy(), torch.cat(lbls).numpy()

def run_tsne(emb, max_samples=10000, random_state=42, base_perp=30):
    n = emb.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, max_samples, replace=False)
        emb = emb[idx]; chosen = idx
    else:
        chosen = np.arange(n)
    emb = StandardScaler().fit_transform(emb)
    # t-SNE zahtijeva perplexity < n_samples; uzmi sigurnu vrijednost
    perp = min(base_perp, max(5, emb.shape[0] // 3 - 1))
    tsne = TSNE(
        n_components=2, init="pca", perplexity=perp,
        learning_rate="auto", n_iter=1000, random_state=random_state, verbose=1
    )
    proj = tsne.fit_transform(emb)
    return proj, chosen

# ---------------------------
#  Palete i crtanje
# ---------------------------
def get_palette(name, n_colors):
    name_l = name.lower()
    if name_l == "nord":
        base = ["#5E81AC","#BF616A","#A3BE8C","#EBCB8B","#88C0D0",
                "#D08770","#B48EAD","#81A1C1","#8FBCBB","#4C566A"]
        if n_colors <= len(base):
            return base[:n_colors]
        reps = (n_colors + len(base) - 1) // len(base)
        return (base * reps)[:n_colors]
    # diskretno uzorkovanje iz ugrađenih kolormapova
    name_map = {"tab10":"tab10", "tab20":"tab20", "set3":"Set3", "hsv":"hsv"}
    cmap = plt.cm.get_cmap(name_map.get(name_l, "tab10"), n_colors)
    return cmap.colors

def make_cmap_norm(n_classes, palette_name):
    colors = get_palette(palette_name, n_classes)
    cmap = ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes)
    return cmap, norm

def save_tsne_bitmap(proj, lbls, out_base,
                     style="mono", page="single", dpi=1000, palette_name="tab10"):
    width_px = 3543 if page == "single" else 7480  # 1000 dpi: single-column / full-page
    size_in = width_px / dpi
    plt.figure(figsize=(size_in, size_in), dpi=dpi)
    ax = plt.gca(); ax.set_axis_off()

    if style == "mono":
        plt.scatter(proj[:, 0], proj[:, 1], s=1, linewidths=0, c="k", alpha=0.80)
    else:
        n_classes = int(lbls.max()) + 1
        cmap, norm = make_cmap_norm(n_classes, palette_name)
        plt.scatter(proj[:, 0], proj[:, 1], s=1, linewidths=0,
                    c=lbls, cmap=cmap, norm=norm, alpha=0.90)

    plt.margins(0.02)
    for ext in ("png", "tiff", "jpg"):
        plt.savefig(f"{out_base}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()

# ---------------------------
#  Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_p1", required=True, help="put do best_phase1.pth")
    ap.add_argument("--ckpt_p2", required=True, help="put do best_phase2.pth")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=10000)
    ap.add_argument("--base_perp", type=int, default=30)
    ap.add_argument("--style", choices=["mono","color"], default="color")
    ap.add_argument("--page",  choices=["single","full"], default="single")
    ap.add_argument("--dpi",   type=int, default=1000)
    ap.add_argument("--out_prefix", default="tsne_resnet18_cifar10")
    ap.add_argument("--palette", default="tab10",
                    choices=["tab10","tab20","set3","nord","hsv"],
                    help="diskretna paleta boja za klase (koristi se kad je --style color)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 test (isti norm kao u treningu)
    mean = (0.4914, 0.4822, 0.4465); std = (0.247, 0.243, 0.261)
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds_test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
    loader  = DataLoader(ds_test, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    # Model
    model = HebbianResNet18(num_classes=10).to(device)

    # ---- Phase 1 ----
    state1 = torch.load(args.ckpt_p1, map_location=device)
    if isinstance(state1, dict) and "state_dict" in state1: state1 = state1["state_dict"]
    if any(k.startswith("module.") for k in state1.keys()):
        state1 = {k.replace("module.","",1): v for k,v in state1.items()}
    model.load_state_dict(state1, strict=False)
    emb1, y = extract_embeddings(model, loader, device)
    proj1, idx1 = run_tsne(emb1, max_samples=args.max_samples, base_perp=args.base_perp)
    save_tsne_bitmap(proj1, y[idx1],
                     f"{args.out_prefix}_phase1_1000dpi",
                     style=args.style, page=args.page, dpi=args.dpi,
                     palette_name=args.palette)

    # ---- Phase 2 ----
    state2 = torch.load(args.ckpt_p2, map_location=device)
    if isinstance(state2, dict) and "state_dict" in state2: state2 = state2["state_dict"]
    if any(k.startswith("module.") for k in state2.keys()):
        state2 = {k.replace("module.","",1): v for k,v in state2.items()}
    model.load_state_dict(state2, strict=False)
    emb2, y = extract_embeddings(model, loader, device)
    proj2, idx2 = run_tsne(emb2, max_samples=args.max_samples, base_perp=args.base_perp)
    save_tsne_bitmap(proj2, y[idx2],
                     f"{args.out_prefix}_phase2_1000dpi",
                     style=args.style, page=args.page, dpi=args.dpi,
                     palette_name=args.palette)

    print("Saved:")
    print(f"  {args.out_prefix}_phase1_1000dpi.(png|tiff|jpg)")
    print(f"  {args.out_prefix}_phase2_1000dpi.(png|tiff|jpg)")

if __name__ == "__main__":
    main()
