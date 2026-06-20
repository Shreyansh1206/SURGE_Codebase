"""Plot Dino + MiniGrid eval scores from block_summaries.json."""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SUMMARIES = "checkpoints_alternate/block_summaries.json"
DEFAULT_OUT = "checkpoints_alternate/block_scores_graph.png"
DEFAULT_MG_OUT = "checkpoints_alternate/block_minigrid_scores_graph.png"
DEFAULT_DINO_OUT = "checkpoints_alternate/block_dino_scores_graph.png"


def load_scores(path: str, max_block: int) -> tuple[list[int], list[float], list[float]]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    blocks = list(range(max_block + 1))
    dino = [math.nan] * len(blocks)
    mg = [math.nan] * len(blocks)

    for row in rows:
        idx = int(row["block_index"])
        if idx > max_block:
            continue
        ev = row.get("eval", {})
        if row["task"] == "dino":
            dino[idx] = float(ev["score_mean"])
        else:
            mg[idx] = float(ev["solve_rate"]) * 100.0

    return blocks, dino, mg


def plot_minigrid_only(
    mg_x: list[int],
    mg_y: list[float],
    max_block: int,
    out_path: str,
) -> None:
    if not mg_x:
        print("[skip] no MiniGrid eval data in range — minigrid graph not written")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        mg_x, mg_y, "s--", color="#0077b6", linewidth=2.5, markersize=9,
        label="MiniGrid solve rate (%)", dashes=(5, 3),
    )

    for bx, by in zip(mg_x, mg_y):
        ax.annotate(
            f"{by:.1f}%",
            (bx, by),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#0077b6",
        )

    ax.set_xlabel("Training block index (MiniGrid blocks only)", fontsize=12)
    ax.set_ylabel("MiniGrid solve rate % (eval)", color="#0077b6", fontsize=12)
    ax.set_title(
        f"MiniGrid DoorKey — eval solve rate (blocks 0–{max_block})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(mg_x)
    ax.set_xlim(min(mg_x) - 1, max(mg_x) + 1)

    y_min, y_max = min(mg_y), max(mg_y)
    pad = max(1.5, (y_max - y_min) * 0.35)
    ax.set_ylim(max(0, y_min - pad), min(100, y_max + pad))

    ax.grid(True, alpha=0.35, linestyle=":")
    ax.legend(loc="lower right")
    ax.tick_params(axis="y", labelcolor="#0077b6")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {os.path.abspath(out_path)}")


def plot_dino_only(
    dino_x: list[int],
    dino_y: list[float],
    max_block: int,
    out_path: str,
) -> None:
    if not dino_x:
        print("[skip] no Dino eval data in range — dino graph not written")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        dino_x, dino_y, "o--", color="#e85d04", linewidth=2.5, markersize=9,
        label="Dino mean score", dashes=(5, 3),
    )

    for bx, by in zip(dino_x, dino_y):
        ax.annotate(
            f"{by:.0f}",
            (bx, by),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#e85d04",
        )

    ax.set_xlabel("Training block index (Dino blocks only)", fontsize=12)
    ax.set_ylabel("Dino mean score (eval)", color="#e85d04", fontsize=12)
    ax.set_title(
        f"Dino runner — eval mean score (blocks 0–{max_block})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(dino_x)
    ax.set_xlim(min(dino_x) - 1, max(dino_x) + 1)

    y_min, y_max = min(dino_y), max(dino_y)
    pad = max(20, (y_max - y_min) * 0.12)
    ax.set_ylim(max(0, y_min - pad), y_max + pad)

    ax.grid(True, alpha=0.35, linestyle=":")
    ax.legend(loc="upper left")
    ax.tick_params(axis="y", labelcolor="#e85d04")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {os.path.abspath(out_path)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", default=DEFAULT_SUMMARIES)
    p.add_argument("--max-block", type=int, default=19)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--mg-out", default=DEFAULT_MG_OUT)
    p.add_argument("--dino-out", default=DEFAULT_DINO_OUT)
    args = p.parse_args()

    blocks, dino, mg = load_scores(args.summaries, args.max_block)
    x = np.array(blocks)

    fig, ax_dino = plt.subplots(figsize=(12, 6))
    ax_mg = ax_dino.twinx()

    dino_x = [i for i in blocks if not math.isnan(dino[i])]
    dino_y = [dino[i] for i in dino_x]
    mg_x = [i for i in blocks if not math.isnan(mg[i])]
    mg_y = [mg[i] for i in mg_x]

    ax_dino.plot(
        dino_x, dino_y, "o--", color="#e85d04", linewidth=2, markersize=7,
        label="Dino mean score", dashes=(4, 3),
    )
    ax_mg.plot(
        mg_x, mg_y, "s--", color="#0077b6", linewidth=2, markersize=7,
        label="MiniGrid solve rate (%)", dashes=(4, 3),
    )

    ax_dino.set_xlabel("Block index", fontsize=12)
    ax_dino.set_ylabel("Dino mean score (eval)", color="#e85d04", fontsize=12)
    ax_mg.set_ylabel("MiniGrid solve rate % (eval)", color="#0077b6", fontsize=12)
    ax_dino.set_title(
        f"Alternating training — eval scores (blocks 0–{args.max_block})",
        fontsize=14,
        fontweight="bold",
    )
    ax_dino.set_xticks(blocks)
    ax_dino.set_xlim(-0.5, args.max_block + 0.5)
    ax_dino.grid(True, alpha=0.3)
    ax_dino.tick_params(axis="y", labelcolor="#e85d04")
    ax_mg.tick_params(axis="y", labelcolor="#0077b6")
    ax_mg.set_ylim(0, 105)

    lines = ax_dino.get_lines() + ax_mg.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax_dino.legend(lines, labels, loc="upper left")

    for i in x:
        task = "MG" if i % 2 == 0 else "Dino"
        ax_dino.annotate(
            task, (i, 0), textcoords="offset points", xytext=(0, -18),
            ha="center", fontsize=7, color="gray",
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {os.path.abspath(args.out)}")

    plot_minigrid_only(mg_x, mg_y, args.max_block, args.mg_out)
    plot_dino_only(dino_x, dino_y, args.max_block, args.dino_out)


if __name__ == "__main__":
    main()
