import os
import glob
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit


# -----------------------
# Model builder
# -----------------------
def build_resnet18(num_classes: int):
    """
    Try importing a custom HebbianResNet18 from train.py; fall back to torchvision resnet18.
    """
    try:
        from train import HebbianResNet18  # rename if your class differs
        m = HebbianResNet18(num_classes=num_classes)
        return m
    except Exception:
        m = models.resnet18(num_classes=num_classes)
        return m


# -----------------------
# Data loaders
# -----------------------
def build_loader(dataset: str = "cifar10",
                 split: str = "val",
                 batch_size: int = 128,
                 num_workers: int = 4):
    assert dataset in {"cifar10", "cifar100"}
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.247, 0.243, 0.261)
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean, std)])

    if dataset == "cifar10":
        if split == "test":
            ds = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
            return DataLoader(ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        full = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
        targets = np.array(full.targets)
    else:  # cifar100
        if split == "test":
            ds = datasets.CIFAR100("./data", train=False, download=True, transform=tfm)
            return DataLoader(ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        full = datasets.CIFAR100("./data", train=True, download=True, transform=tfm)
        targets = np.array(full.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=10_000, random_state=42)
    _, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    val = Subset(full, val_idx.tolist())
    return DataLoader(val, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# -----------------------
# Hook selection for ResNet-18
# -----------------------
def get_hebb_module_resnet18(model: nn.Module, post_activation: bool = True):
    """
    Select the module to hook in layer2 last block.
    post_activation=True  -> hook bn2 and apply ReLU inside the hook
    post_activation=False -> hook conv2 (pre-activation)
    """
    block = model.layer2[-1]
    if post_activation:
        return block.bn2
    else:
        return block.conv2


# -----------------------
# HAF computation
# -----------------------
@torch.no_grad()
def compute_haf_from_module(model,
                            module_to_hook,
                            dataloader,
                            device="cuda",
                            tau=0.8,
                            post_activation=True,
                            use_percentile=None,
                            debug=False):
    """
    Compute per-channel HAF for a given module.

    tau: threshold fraction (0..1)
    post_activation: if True, apply ReLU to hook output
    use_percentile: if set (e.g., 99), use that percentile across images per channel
                    instead of the strict per-channel max as the reference peak
    """
    model.eval()
    peaks = []

    def hook(_, __, out):
        out = torch.relu(out) if post_activation else out
        # out: [N, C, H, W]
        peak = out.flatten(2).amax(dim=2)  # [N, C]
        peaks.append(peak.detach().cpu())

    h = module_to_hook.register_forward_hook(hook)

    total = 0
    for x, _ in dataloader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        total += x.size(0)

    h.remove()

    if not peaks:
        return np.array([]), float("nan"), total

    alphas = torch.cat(peaks, 0)  # [N, C]

    if use_percentile is not None:
        # robust per-channel reference using percentile
        p = np.percentile(alphas.numpy(), use_percentile, axis=0)  # [C]
        alpha_ref = torch.from_numpy(p).float()
    else:
        alpha_ref = alphas.max(dim=0).values  # [C]

    if debug:
        print("[debug] alphas shape:", tuple(alphas.shape))
        print("[debug] alpha_ref stats: mean={:.6f} med={:.6f} min={:.6f} max={:.6f}".format(
            alpha_ref.mean().item(),
            alpha_ref.median().item(),
            alpha_ref.min().item(),
            alpha_ref.max().item(),
        ))

    haf = torch.zeros_like(alpha_ref, dtype=torch.float32)
    live = alpha_ref > 0
    if live.any():
        thresh = tau * alpha_ref[live]            # [C_live]
        hits = (alphas[:, live] >= thresh).float()  # [N, C_live]
        haf[live] = hits.mean(dim=0)

        if debug:
            print("[debug] thresh stats: mean={:.6f} med={:.6f}".format(
                thresh.mean().item(), thresh.median().item()))
            print("[debug] per-ch HAF stats: mean={:.6f} med={:.6f} min={:.6f} max={:.6f}".format(
                hits.mean(dim=0).mean().item(),
                hits.mean(dim=0).median().item(),
                hits.mean(dim=0).min().item(),
                hits.mean(dim=0).max().item(),
            ))

    mean_haf = float(haf.mean().item())
    return haf.numpy(), mean_haf, total


# -----------------------
# Runner
# -----------------------
def run_haf(base_dir="results_exp_resnet18",
            split="val",
            dataset="cifar10",
            taus=(0.8,),
            device="cuda",
            out_csv="haf_summary_resnet18.csv",
            post_activation=True,
            percentile=None,
            debug=False):
    os.makedirs(base_dir, exist_ok=True)
    loader = build_loader(dataset=dataset, split=split)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    run_dirs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if not run_dirs:
        # if there are no run_* folders, treat base_dir as a single run directory
        run_dirs = [base_dir]

    rows = []
    num_classes = 10 if dataset == "cifar10" else 100

    for run_dir in run_dirs:
        for phase_name, ckpt in [("phase1", "best_phase1.pth"), ("phase2", "best_phase2.pth")]:
            ckpt_path = os.path.join(run_dir, ckpt)
            if not os.path.isfile(ckpt_path):
                print(f"Skip: {ckpt_path} not found")
                continue

            # build and load model
            model = build_resnet18(num_classes=num_classes).to(device)
            state = torch.load(ckpt_path, map_location=device)

            # support checkpoints with "state_dict" key
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # strip DataParallel "module." if present
            if isinstance(state, dict) and any(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}

            model.load_state_dict(state, strict=False)

            module = get_hebb_module_resnet18(model, post_activation=post_activation)

            for tau in taus:
                haf_vec, mean_haf, n_imgs = compute_haf_from_module(
                    model, module, loader, device=device, tau=tau,
                    post_activation=post_activation, use_percentile=percentile, debug=debug
                )

                # save per-filter vector
                npy_path = os.path.join(run_dir, f"haf_{phase_name}_{dataset}_{split}_tau{tau:.2f}.npy")
                np.save(npy_path, haf_vec)

                rows.append({
                    "run": os.path.basename(run_dir.rstrip("/")),
                    "phase": phase_name,
                    "dataset": dataset,
                    "split": split,
                    "tau": tau,
                    "post_activation": int(post_activation),
                    "percentile": percentile if percentile is not None else "",
                    "n_images": n_imgs,
                    "mean_haf": round(float("nan") if (haf_vec is None or len(haf_vec) == 0) else float(np.mean(haf_vec)), 6),
                    "n_filters": 0 if haf_vec is None else int(haf_vec.shape[0]),
                    "per_filter_path": npy_path
                })
                print(f"{run_dir} | {phase_name} | {dataset}/{split} | tau={tau:.2f} | "
                      f"mean HAF={np.mean(haf_vec) if haf_vec is not None and len(haf_vec)>0 else float('nan'):.3f} "
                      f"over {n_imgs} images")

    # write CSV
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"> Saved summary to {out_csv}")
    else:
        print("No rows written (check checkpoint paths).")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="results_exp_resnet18")
    parser.add_argument("--dataset",  default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--split",    default="val", choices=["val", "test"])
    parser.add_argument("--taus",     type=float, nargs="+", default=[0.8], help="One or more tau values")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--out_csv",  default="haf_summary_resnet18.csv")
    parser.add_argument("--pre_act",  action="store_true", help="Hook pre-activation (Conv2d) instead of post-activation")
    parser.add_argument("--percentile", type=float, default=None,
                        help="If set, use this percentile (e.g., 99) as per-channel reference instead of max")
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args()

    run_haf(
        base_dir=args.base_dir,
        split=args.split,
        dataset=args.dataset,
        taus=tuple(args.taus),
        device=args.device,
        out_csv=args.out_csv,
        post_activation=not args.pre_act,
        percentile=args.percentile,
        debug=args.debug
    )

