"""Augment per-frame nav-graph JSONs with ExploRFM trav/frontier scores.

For every `graph_NNNNNN.json` in the dataset, this loads the matching
`rgb_NNNNNN.png`, runs ExploRFM inference, samples the predicted
traversability and frontier maps at each node's pixel, and writes a
mirrored JSON with two new fields per node:

    "trav_score":     float in [0, 1]
    "frontier_score": float in [0, 1]

Output mirrors the dataset directory structure under --output-root.

Example:
    python explorfm/generate_cleaned_jsons.py \
        --dataset-root "/home/rohang73/Desktop/longrange" \
        --output-root  "/home/rohang73/Desktop/longrange_explorfm_clean" \
        --ckpts-dir    "/home/rohang73/Desktop/rgn_graph_cleaning/nebula2-wildos/ckpts"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explorfm.explorfm_model import ExploRFMInference  # noqa: E402


def build_model(ckpts_dir: Path, device: str) -> ExploRFMInference:
    frontier_ckpt = str(ckpts_dir / "frontier_head.ckpt")
    trav_ckpt = str(ckpts_dir / "trav_head.ckpt")
    siglip2_dir = str(ckpts_dir / "siglip2")
    local_radio = ckpts_dir / "c-radio_v3-b_half.pth.tar"
    radio_version = str(local_radio) if local_radio.exists() else "c-radio_v3-b"
    print(f"[init] radio backbone: {radio_version}")

    return ExploRFMInference(
        frontier_ckpt=frontier_ckpt,
        traversability_ckpt=trav_ckpt,
        model_version=radio_version,
        adaptor_version="siglip2",
        adaptor_ckpt_path=siglip2_dir,
        use_naclip=True,
        use_summary_for_spatial=True,
        radio_dim=768,
        static_scale_factor=0.5,
        model_precision="FP32",
        device=device,
    )


def sample_scores(
    trav_np: np.ndarray, front_np: np.ndarray, px: float, py: float
) -> tuple[float | None, float | None]:
    H, W = trav_np.shape
    x, y = int(round(px)), int(round(py))
    if 0 <= x < W and 0 <= y < H:
        return float(trav_np[y, x]), float(front_np[y, x])
    return None, None


def process_frame(
    model: ExploRFMInference,
    rgb_path: Path,
    graph_path: Path,
    out_path: Path,
) -> dict:
    img_np = np.array(Image.open(rgb_path).convert("RGB"))
    with torch.inference_mode():
        trav, front, _ = model.forward_on_numpy(img_np)
    trav_np = trav.squeeze().float().cpu().numpy()
    front_np = front.squeeze().float().cpu().numpy()

    with open(graph_path) as f:
        data = json.load(f)

    n_nodes = 0
    n_oob = 0
    for node in data.get("nodes", []):
        px, py = node["pixel"]
        t, fr = sample_scores(trav_np, front_np, px, py)
        node["trav_score"] = t
        node["frontier_score"] = fr
        n_nodes += 1
        if t is None:
            n_oob += 1

    data["explorfm_inference"] = {
        "rgb_file": rgb_path.name,
        "image_size": [int(img_np.shape[1]), int(img_np.shape[0])],
        "score_map_size": [int(trav_np.shape[1]), int(trav_np.shape[0])],
        "n_nodes_scored": n_nodes,
        "n_nodes_out_of_bounds": n_oob,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(out_path)

    return {"nodes": n_nodes, "oob": n_oob}


def iter_jobs(
    dataset_root: Path, output_root: Path, force: bool
) -> list[tuple[Path, Path, Path]]:
    jobs: list[tuple[Path, Path, Path]] = []
    skipped_no_rgb = 0
    skipped_existing = 0
    for mission_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for graph_path in sorted(mission_dir.glob("graph_*.json")):
            idx = graph_path.stem.replace("graph_", "")
            rgb_path = mission_dir / f"rgb_{idx}.png"
            if not rgb_path.exists():
                skipped_no_rgb += 1
                continue
            out_path = output_root / mission_dir.name / graph_path.name
            if out_path.exists() and not force:
                skipped_existing += 1
                continue
            jobs.append((rgb_path, graph_path, out_path))
    print(
        f"[scan] {len(jobs)} frames to process    "
        f"(skipped: {skipped_existing} already-done, {skipped_no_rgb} missing rgb)"
    )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/rohang73/Desktop/ASL RGB NAV GRAPH"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/rohang73/Desktop/ASL RGB NAV GRAPH cleaned"),
    )
    parser.add_argument(
        "--ckpts-dir",
        type=Path,
        default=REPO_ROOT / "ckpts",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess frames even if the output JSON already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many frames (debug aid).",
    )
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        raise SystemExit(f"dataset root does not exist: {args.dataset_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[paths] dataset: {args.dataset_root}")
    print(f"[paths] output:  {args.output_root}")
    print(f"[paths] ckpts:   {args.ckpts_dir}")
    print(f"[paths] device:  {args.device}")

    jobs = iter_jobs(args.dataset_root, args.output_root, args.force)
    if args.limit is not None:
        jobs = jobs[: args.limit]
        print(f"[limit] truncated to {len(jobs)} frames")
    if not jobs:
        print("[done] nothing to do")
        return

    model = build_model(args.ckpts_dir, args.device)

    t0 = time.time()
    total_nodes = 0
    total_oob = 0
    for i, (rgb_path, graph_path, out_path) in enumerate(jobs, start=1):
        try:
            stats = process_frame(model, rgb_path, graph_path, out_path)
        except Exception as e:
            print(f"[err]  {graph_path.name} ({rgb_path.parent.name}): {e}")
            continue
        total_nodes += stats["nodes"]
        total_oob += stats["oob"]
        if i == 1 or i % 25 == 0 or i == len(jobs):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-9)
            eta = (len(jobs) - i) / max(rate, 1e-9)
            print(
                f"[{i:5d}/{len(jobs)}]  {rgb_path.parent.name}/{rgb_path.name}    "
                f"nodes={stats['nodes']:3d} oob={stats['oob']:2d}    "
                f"{rate:.2f} it/s    eta={eta/60:.1f} min"
            )

    print(
        f"[done] {len(jobs)} frames    "
        f"{total_nodes} nodes scored    {total_oob} out-of-bounds    "
        f"in {(time.time() - t0)/60:.1f} min"
    )


if __name__ == "__main__":
    main()
