# src/prune.py
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from pathlib import Path
import argparse
import copy
from tqdm import tqdm
import numpy as np

from src.models import get_model, get_device
from src.train import load_checkpoint  # optional reuse of loader if you added it else implement simple loader

# -------------------------
# Utilities
# -------------------------
def print_sparsity(model):
    total_zeros = 0
    total_elems = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            nz = int((param == 0).sum().item())
            total = param.numel()
            total_zeros += nz
            total_elems += total
            print(f"{name}: zeros {nz}/{total} ({100.0 * nz / total:.2f}%)")
    if total_elems > 0:
        print(f"Overall sparsity: {100.0 * total_zeros / total_elems:.2f}%")
    return total_zeros, total_elems

def remove_pruning_reparametrization(model):
    # After pruning, remove reparam to have static weights (prune.remove)
    for module in model.modules():
        # prune module only if it has been pruned
        try:
            # common patterns: 'weight_mask' attribute presence
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
        except Exception:
            pass

# -------------------------
# Pruning functions
# -------------------------
def global_unstructured_prune(model, amount=0.2, prune_fn=prune.L1Unstructured):
    """
    Globally prune `amount` fraction of weights using prune_fn (l1 by default).
    Applied only to conv and linear layers.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune_fn, amount=amount)
    return model

def structured_channel_prune(model, amount=0.2, dim=0, n=2):
    """
    Structured pruning (prunes entire channels/filters) using ln_structured on conv/linear layers.
    dim=0 means prune output channels (filters) for Conv2d.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
        if isinstance(module, nn.Linear):
            # for linear, structured pruning may not be as meaningful; skip or prune rows
            prune.ln_structured(module, name='weight', amount=amount, n=n, dim=0)
    return model

def one_shot_prune_and_save(weights_in, weights_out, model_name="mobilenet_v2", amount=0.5, structured=False):
    device = get_device()
    model = get_model(model_name, num_classes=2, pretrained=False)
    state = torch.load(weights_in, map_location='cpu')
    # try to unwrap state dict if checkpoint
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # remove module. prefix if present
    new_state = {}
    for k,v in state.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    model = model.to(device)
    if structured:
        structured_channel_prune(model, amount=amount)
    else:
        global_unstructured_prune(model, amount=amount)
    print("Sparsity after pruning:")
    print_sparsity(model)
    remove_pruning_reparametrization(model)
    torch.save(model.state_dict(), weights_out)
    print(f"Saved pruned model to {weights_out}")


# -------------------------
# Iterative Magnitude Pruning (IMP)
# -------------------------
def iterative_prune(
    weights_in,
    out_dir,
    model_name="mobilenet_v2",
    rounds=5,
    prune_per_round=0.2,
    finetune_epochs=2,
    finetune_fn=None,   # signature: fn(model, epochs, out_dir, round_idx)
    device=None
):
    """
    Iteratively prune and call finetune function between rounds.
     - prune_per_round: fraction to remove each round (e.g., 0.2 -> 20% of remaining)
     - rounds: number of rounds
    finetune_fn must be provided; it should fine-tune model and return path to best weights for next round.
    """
    device = device or get_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load initial model
    model = get_model(model_name, num_classes=2, pretrained=False)
    state = torch.load(weights_in, map_location='cpu')
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # fix keys
    new_state = {}
    for k,v in state.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)

    current_weights = weights_in
    cumulative_amount = 0.0
    for r in range(rounds):
        print(f"\n=== IMP Round {r+1}/{rounds} ===")
        # compute amount to prune globally this round such that overall fraction removed equals target?
        # Here prune_per_round applies to remaining weights; cumulative sparsity increases multiplicatively.
        amount = prune_per_round
        print(f"Pruning {amount*100:.1f}% of remaining weights (round {r+1})")
        # apply pruning to model loaded from current_weights
        model = get_model(model_name, num_classes=2, pretrained=False)
        st = torch.load(current_weights, map_location='cpu')
        if isinstance(st, dict) and "state_dict" in st:
            st = st["state_dict"]
        # fix keys
        ns = {}
        for k,v in st.items():
            nk = k
            if k.startswith("module."):
                nk = k[len("module."):]
            ns[nk] = v
        model.load_state_dict(ns, strict=False)
        # apply global prune
        global_unstructured_prune(model, amount=amount)
        print("Sparsity after pruning (raw):")
        print_sparsity(model)
        # remove reparam so finetuning works on the pruned weights (weights remain zeroed)
        remove_pruning_reparametrization(model)
        pruned_weights_path = out_dir / f"pruned_round{r+1}.pth"
        torch.save(model.state_dict(), str(pruned_weights_path))
        print(f"Saved pruned checkpoint to {pruned_weights_path}")

        # FINETUNE: call user-supplied finetune function to retrain the pruned model
        if finetune_fn is None:
            raise ValueError("finetune_fn must be provided for iterative_prune")
        print(f"Finetuning for {finetune_epochs} epochs (round {r+1})...")
        best_after_finetune = finetune_fn(str(pruned_weights_path), out_dir=str(out_dir), round_idx=r, epochs=finetune_epochs)
        print("Finetune returned:", best_after_finetune)
        current_weights = best_after_finetune

    print("IMP completed. Final weights at:", current_weights)
    final_state = torch.load(current_weights, map_location='cpu')
    # save final state as final_pruned.pth
    final_out = out_dir / "final_pruned.pth"
    torch.save(final_state, final_out)
    print("Saved final pruned model:", final_out)
    return str(final_out)


# -------------------------
# Example finetune_fn scaffold (user implement)
# -------------------------
def example_finetune_fn(weights_path, out_dir, round_idx=0, epochs=2):
    """
    This is a scaffold. You should implement fine-tuning logic (call your train script or API).
    For example, call train.py with a config that loads weights_path as resume and finetune for N epochs.
    As a simple example we return the same weights_path (NOT useful). Replace with real training.
    """
    # TODO: call training script via subprocess or import training loop and run a few epochs here.
    # e.g. subprocess.run(["python", "src/train.py", "--config", "configs/mobilenet_finetune.yaml", "--resume", weights_path])
    return weights_path


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["one_shot","imp"], default="one_shot")
    parser.add_argument("--weights_in", required=True)
    parser.add_argument("--weights_out", default=None)
    parser.add_argument("--model", default="mobilenet_v2")
    parser.add_argument("--amount", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--prune_per_round", type=float, default=0.2)
    parser.add_argument("--finetune_epochs", type=int, default=2)
    parser.add_argument("--out_dir", default="checkpoints/pruned")
    args = parser.parse_args()

    if args.mode == "one_shot":
        out = args.weights_out or Path(args.out_dir) / "pruned_one_shot.pth"
        one_shot_prune_and_save(args.weights_in, str(out), model_name=args.model, amount=args.amount, structured=False)
    else:
        # For iterative mode, you must implement finetune_fn above or pass a callback
        print("Running IMP (iterative magnitude pruning). NOTE: implement finetune_fn in this script to run full loop.")
        iterative_prune(args.weights_in, args.out_dir, model_name=args.model, rounds=args.rounds, prune_per_round=args.prune_per_round, finetune_epochs=args.finetune_epochs)

# comamnd to run this script to prune 60% of weight: python src/prune.py --mode one_shot --weights_in checkpoints/mobilenet_distill/model_best.pth.tar --weights_out checkpoints/pruned/pruned_60.pth --amount 0.6 --model mobilenet_v2
# command to run iterative pruning: python src/prune.py --mode imp --weights_in checkpoints/mobilenet_distill/model_best.pth.tar --out_dir checkpoints/pruned/imp --rounds 5 --prune_per_round 0.2 --finetune_epochs 2
