import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch

from config import DATASET_INFO, DISTILL_EPOCHS, PHASE_MAX_EPOCHS, ROUNDS, SEEDS, THRESHOLDS
from data import get_data, make_loader, seed_everything
from model import build_model
from training import evaluate, run_mode, train_local


def summarize(per_round_path, out_dir):
    per_round = pd.read_csv(per_round_path)
    best_rows = []
    for (dataset, mode, seed), group in per_round[per_round["round"] > 0].groupby(
        ["dataset", "mode", "seed"], sort=True
    ):
        best = group.loc[group["server_acc"].idxmax()]
        reached = group[group["server_acc"] >= THRESHOLDS[dataset]]
        best_rows.append({
            "dataset": dataset,
            "model": "convnidd" if dataset == "cifar10" else "net",
            "mode": mode,
            "seed": seed,
            "best_server_acc": best["server_acc"], "best_round": int(best["round"]),
            "client_mean_acc_at_best_round": best["client_mean_acc"],
            "rounds_to_threshold": reached["round"].min() if len(reached) else np.nan,
        })
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(out_dir, "results_best_per_seed.csv"), index=False)
    final = per_round[per_round["round"] == per_round["round"].max()]
    summary = best_df.groupby(["dataset", "model", "mode"], as_index=False).agg(
        best_server_acc_mean=("best_server_acc", "mean"),
        best_server_acc_std=("best_server_acc", "std"),
        best_round_mean=("best_round", "mean"),
        client_mean_acc_at_best_round_mean=("client_mean_acc_at_best_round", "mean"),
        rounds_to_threshold_mean=("rounds_to_threshold", "mean"),
    )
    final_summary = final.groupby(["dataset", "model", "mode"], as_index=False).agg(
        final_server_acc_mean=("server_acc", "mean"), final_server_acc_std=("server_acc", "std")
    )
    summary = summary.merge(final_summary, on=["dataset", "model", "mode"])[
        ["dataset", "model", "mode", "best_server_acc_mean", "best_server_acc_std",
         "best_round_mean", "client_mean_acc_at_best_round_mean", "final_server_acc_mean",
         "final_server_acc_std", "rounds_to_threshold_mean"]
    ]
    summary.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_INFO), default=list(DATASET_INFO))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--out", default="results")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="preserve existing results and skip completed dataset/seed combinations",
    )
    args = parser.parse_args()
    rounds, distill_epochs, phase_max = args.rounds, DISTILL_EPOCHS, PHASE_MAX_EPOCHS
    seeds = args.seeds
    if args.quick:
        seeds, rounds, distill_epochs, phase_max = [seeds[0]], 2, 1, 2
    os.makedirs(args.out, exist_ok=True)
    per_round_path = os.path.join(args.out, "results_per_round.csv")
    completed = set()
    if args.resume and os.path.exists(per_round_path):
        existing = pd.read_csv(per_round_path)
        expected_rounds = set(range(rounds + 1))
        completed = {
            (dataset, seed, mode)
            for (dataset, seed, mode), group in existing.groupby(
                ["dataset", "seed", "mode"]
            )
            if set(group["round"]) >= expected_rounds
        }
    elif os.path.exists(per_round_path):
        os.remove(per_round_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    header = ["dataset", "model", "mode", "seed", "round", "server_acc",
              "client_mean_acc", "client_std_acc"]

    for dataset_name in args.datasets:
        for seed in seeds:
            if all((dataset_name, seed, mode) in completed
                   for mode in ("c2s_only", "bidirectional")):
                print(dataset_name, seed, "already complete; skipping")
                continue
            seed_everything(seed)
            _, test, ref_dataset, private_sets, input_shape = get_data(
                dataset_name, seed, root=args.data_root
            )
            print(dataset_name, seed, "private sizes:", [len(s) for s in private_sets])
            targets = np.asarray(getattr(private_sets[0].dataset, "targets"))
            for i, private_set in enumerate(private_sets):
                hist = np.bincount(targets[private_set.indices], minlength=10).tolist()
                print(" client", i, "histogram:", hist)
            test_loader = make_loader(test, False)
            phase_clients = [
                build_model(dataset_name, input_shape).to(device) for _ in private_sets
            ]
            phase_states = []
            for client, private_set in zip(phase_clients, private_sets):
                train_local(client, private_set, device, phase_max, early_stop=True)
                phase_states.append(copy.deepcopy(client.state_dict()))
            phase_acc = [evaluate(client, test_loader, device) for client in phase_clients]
            print("Phase I mean client test accuracy:", np.mean(phase_acc))
            global_state = copy.deepcopy(
                build_model(dataset_name, input_shape).to(device).state_dict()
            )
            weights = (np.ones(len(private_sets), dtype=np.float32) / len(private_sets)).tolist()
            for mode in ("c2s_only", "bidirectional"):
                if (dataset_name, seed, mode) in completed:
                    print(dataset_name, seed, mode, "already complete; skipping")
                    continue
                rows = run_mode(
                    mode, phase_states, global_state, private_sets, ref_dataset, test_loader,
                    dataset_name, input_shape, weights, seed, device, args.alpha, rounds,
                    distill_epochs
                )
                frame = pd.DataFrame(
                    [[dataset_name, "convnidd" if dataset_name == "cifar10" else "net"] + row
                     for row in rows],
                    columns=header,
                )
                frame[["server_acc", "client_mean_acc", "client_std_acc"]] = frame[
                    ["server_acc", "client_mean_acc", "client_std_acc"]
                ].round(2)
                frame.to_csv(
                    per_round_path, mode="a",
                    header=not os.path.exists(per_round_path), index=False
                )
    summarize(per_round_path, args.out)


if __name__ == "__main__":
    main()
