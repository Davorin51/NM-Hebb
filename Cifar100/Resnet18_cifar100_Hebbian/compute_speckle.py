# compute_speckle.py
import os, glob, csv, argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# -----------------------------
# Model: CIFAR-style ResNet-18
# -----------------------------
def build_resnet18_cifar(num_classes: int):
    """
    Torchvision ResNet-18 s CIFAR stemom (3x3 conv, stride=1, bez maxpool).
    Ovo se dobro slaže s većinom CIFAR checkpointa (3x3 conv1).
    """
    m = models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

def get_weight_module_resnet18(model: nn.Module, which: str = "layer2_last_conv2") -> nn.Conv2d:
    """
    Odabir sloja čije kernele analiziramo (po defaultu: zadnji BasicBlock u layer2, conv2).
    """
    if which == "layer2_last_conv2":
        return model.layer2[-1].conv2
    elif which == "layer2_first_conv2":
        return model.layer2[0].conv2
    elif which == "layer1_first_conv2":
        return model.layer1[0].conv2
    else:
        raise ValueError(f"Unknown layer selector: {which}")

# ---------------------------------------
# Speckle metrika na temelju FFT spektra
# ---------------------------------------
def _radial_orientation_grids(k: int, eps: float = 1e-9):
    """Pripremi mreže za radijus i kut na KxK frekvencijskoj rešetki (FFT shift)."""
    cy = (k - 1) / 2.0
    cx = (k - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    y = yy - cy
    x = xx - cx
    r = np.sqrt(x * x + y * y)
    r_max = r.max() + eps
    r_norm = r / r_max
    theta = np.arctan2(y, x)  # [-pi, pi]
    return r_norm, theta

def per_filter_speckle_scores(W: np.ndarray, rho: float = 0.5, nbins: int = 36):
    """
    W: težine konvolucijskog sloja, shape [C_out, C_in, K, K] (numpy).
    Vraća:
      - E_hf:  high-frequency energy fraction po filtru, shape [C_out]
      - R:     orientation concentration po filtru,   shape [C_out]
    """
    C_out, C_in, K, _ = W.shape
    r_norm, theta = _radial_orientation_grids(K)
    annulus = r_norm >= rho

    # priprema binova za orijentacije (periodičnost pi -> koristimo 2*theta)
    thetas = np.linspace(-np.pi, np.pi, nbins + 1)
    theta_centers = 0.5 * (thetas[:-1] + thetas[1:])

    E_hf = np.zeros(C_out, dtype=np.float64)
    R = np.zeros(C_out, dtype=np.float64)

    for f in range(C_out):
        w = W[f].mean(axis=0)  # [K, K], prosjek po input kanalima
        w = w - w.mean()
        std = w.std() + 1e-8
        w = w / std

        F = np.fft.fftshift(np.fft.fft2(w))
        P = (np.abs(F) ** 2).astype(np.float64)

        total = P.sum() + 1e-12
        hf = P[annulus].sum()
        E_hf[f] = float(hf / total)

        # orijentacijska histograma po annulus zoni
        H = np.zeros(nbins, dtype=np.float64)
        t = theta[annulus]
        p = P[annulus]
        # binning
        inds = np.digitize(t, thetas) - 1
        inds = np.clip(inds, 0, nbins - 1)
        np.add.at(H, inds, p)

        # resultant s dvostrukim kutom (perioda pi)
        vec = np.sum(H * np.exp(1j * 2.0 * theta_centers))
        R[f] = np.abs(vec) / (H.sum() + 1e-12)

    return E_hf, R

def speckle_mask(E_hf: np.ndarray, R: np.ndarray, eta: float = 0.60, kappa: float = 0.20):
    """
    Speckle ako: high-freq udio >= eta i orijentacijska koncentracija <= kappa.
    """
    return (E_hf >= eta) & (R <= kappa)

# -------------------------
# Petlja preko run_* foldera
# -------------------------
def run_speckle(
    base_dir="results_exp_resnet18",
    dataset="cifar10",
    out_csv="speckle_summary_resnet18.csv",
    which_layer="layer2_last_conv2",
    rho=0.5,
    eta=0.60,
    kappa=0.20,
):
    os.makedirs(base_dir, exist_ok=True)
    rows = []
    num_classes = 10 if dataset == "cifar10" else 100

    run_dirs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if not run_dirs:
        run_dirs = [base_dir]

    for run_dir in run_dirs:
        for phase_name, ckpt in [("phase1", "best_phase1.pth"), ("phase2", "best_phase2.pth")]:
            ckpt_path = os.path.join(run_dir, ckpt)
            if not os.path.isfile(ckpt_path):
                print(f"Skip: {ckpt_path} not found")
                continue

            # model i učitavanje težina (strict=False tolerira razlike u headu)
            model = build_resnet18_cifar(num_classes=num_classes)
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and any(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model.eval()

            # odaberi sloj i izvuci težine
            mod = get_weight_module_resnet18(model, which=which_layer)
            W = mod.weight.detach().cpu().numpy()  # [C_out, C_in, K, K]

            # izračun metrika
            E_hf, R = per_filter_speckle_scores(W, rho=rho, nbins=36)
            mask = speckle_mask(E_hf, R, eta=eta, kappa=kappa)
            speckle_pct = 100.0 * mask.mean()
            mean_Ehf = float(E_hf.mean())
            mean_R = float(R.mean())

            # spremi vektore po filtru
            np.save(os.path.join(run_dir, f"speckle_Ehf_{phase_name}.npy"), E_hf)
            np.save(os.path.join(run_dir, f"speckle_R_{phase_name}.npy"), R)
            np.save(os.path.join(run_dir, f"speckle_mask_{phase_name}.npy"), mask.astype(np.uint8))

            print(f"{run_dir} | {phase_name} | speckle={speckle_pct:.1f}%  (Ehf̄={mean_Ehf:.3f}, R̄={mean_R:.3f})")

            rows.append({
                "run": os.path.basename(run_dir.rstrip("/")),
                "phase": phase_name,
                "dataset": dataset,
                "layer": which_layer,
                "rho": rho,
                "eta": eta,
                "kappa": kappa,
                "n_filters": int(W.shape[0]),
                "speckle_pct": round(speckle_pct, 3),
                "Ehf_mean": round(mean_Ehf, 6),
                "R_mean": round(mean_R, 6),
                "Ehf_path": os.path.join(run_dir, f"speckle_Ehf_{phase_name}.npy"),
                "R_path": os.path.join(run_dir, f"speckle_R_{phase_name}.npy"),
                "mask_path": os.path.join(run_dir, f"speckle_mask_{phase_name}.npy"),
            })

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"> Saved summary to {out_csv}")
    else:
        print("No rows written (check checkpoint paths).")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="results_exp_resnet18")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--out_csv", default="speckle_summary_resnet18.csv")
    ap.add_argument("--layer", default="layer2_last_conv2",
                    choices=["layer2_last_conv2", "layer2_first_conv2", "layer1_first_conv2"])
    ap.add_argument("--rho", type=float, default=0.5, help="inner radius (0..1) for HF annulus")
    ap.add_argument("--eta", type=float, default=0.60, help="HF energy threshold for speckle")
    ap.add_argument("--kappa", type=float, default=0.20, help="orientation concentration threshold for speckle")
    args = ap.parse_args()

    run_speckle(
        base_dir=args.base_dir,
        dataset=args.dataset,
        out_csv=args.out_csv,
        which_layer=args.layer,
        rho=args.rho,
        eta=args.eta,
        kappa=args.kappa,
    )

