#!/usr/bin/env python3
"""UMAP/t-SNE visualization of cross-receiver embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--path", default="fused", choices=["main", "oob", "fused"])
    p.add_argument("--method", default="umap", choices=["umap", "tsne"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--level", default="file", choices=["file", "window"])
    return p.parse_args()


def reduce(x: np.ndarray, method: str) -> np.ndarray:
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_neighbors=min(15, len(x) - 1), min_dist=0.1, random_state=0).fit_transform(x)
        except ImportError:
            method = "tsne"
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=min(30, len(x) - 1), random_state=0, init="pca").fit_transform(x)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = "file_embeddings.npz" if args.level == "file" else "window_embeddings.npz"
    data = np.load(Path(args.emb_dir) / fname, allow_pickle=True)
    z = data[args.path]
    labels = data["labels"]
    receivers = data["receivers"]

    xy = reduce(z, args.method)
    rx_names = {1: "RX1", 2: "RX2"}

    # color by receiver
    fig, ax = plt.subplots(figsize=(6, 5))
    for rx in sorted(set(receivers.tolist())):
        mask = receivers == rx
        ax.scatter(xy[mask, 0], xy[mask, 1], s=40, alpha=0.8, label=rx_names.get(rx, f"RX{rx}"))
    ax.set_title(f"{args.method.upper()} — color by receiver ({args.path})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.method}_color_receiver_{args.path}.png", dpi=150)
    fig.savefig(out_dir / f"{args.method}_color_receiver_{args.path}.pdf")
    plt.close(fig)

    # color by device
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=40, cmap="tab20", alpha=0.85)
    ax.set_title(f"{args.method.upper()} — color by device ({args.path})")
    fig.colorbar(sc, ax=ax, label="device label")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.method}_color_device_{args.path}.png", dpi=150)
    fig.savefig(out_dir / f"{args.method}_color_device_{args.path}.pdf")
    plt.close(fig)

    # shape by receiver, color by device
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {1: "o", 2: "^"}
    for rx in sorted(set(receivers.tolist())):
        mask = receivers == rx
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=labels[mask], s=50, cmap="tab20",
            marker=markers.get(rx, "o"),
            edgecolors="k", linewidths=0.3,
            alpha=0.85, label=rx_names.get(rx, f"RX{rx}"),
        )
    ax.set_title(f"{args.method.upper()} — shape=receiver, color=device ({args.path})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.method}_shape_receiver_color_device_{args.path}.png", dpi=150)
    fig.savefig(out_dir / f"{args.method}_shape_receiver_color_device_{args.path}.pdf")
    plt.close(fig)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
