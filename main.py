"""
ALOHA Primitive Skill Decomposer
=================================
Automatically segments ALOHA robot episodes into object-centric
primitive skills: reach / grasp / move / release.

Usage
-----
# Decompose 5 episodes of the cube-transfer dataset (default)
python main.py

# Custom dataset and episode count
python main.py --dataset lerobot/aloha_sim_insertion_human --episodes 10

# Save results without visualisation
python main.py --no-visualize --output results/

# Use locally cached dataset
python main.py --local-root /data/aloha_transfer

Full option list
----------------
  --dataset        HuggingFace repo id  (default: lerobot/aloha_sim_transfer_cube_human)
  --episodes       Number of episodes to process  (default: 5)
  --output         Output directory  (default: outputs/)
  --no-visualize   Skip PNG generation
  --no-json        Skip JSON output
  --local-root     Path to local dataset root
  --smooth-window  Gripper smoothing kernel half-width in frames  (default: 7)
  --fps            Override fps  (default: read from dataset metadata)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm

from config import ALOHAConfig
from src.data.loader import EpisodeLoader
from src.decomposition.segmenter import PrimitiveSegmenter
from src.decomposition.primitives import PrimitiveType
from src.visualization.visualizer import visualize_episode, visualize_summary, visualize_ee_pose
from src.visualization.video_exporter import VideoExporter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ALOHA primitive skill decomposer")
    p.add_argument("--dataset",       default="lerobot/aloha_sim_transfer_cube_human")
    p.add_argument("--episodes",      type=int,   default=5)
    p.add_argument("--output",        default="outputs/")
    p.add_argument("--no-visualize",  action="store_true")
    p.add_argument("--no-video",      action="store_true")
    p.add_argument("--no-json",       action="store_true")
    p.add_argument("--local-root",    default=None)
    p.add_argument("--smooth-window", type=int,   default=7)
    p.add_argument("--fps",           type=int,   default=None)

    args = p.parse_args()

    cfg = ALOHAConfig(
        dataset_repo_id = args.dataset,
        n_episodes      = args.episodes,
        output_dir      = args.output,
        local_root      = args.local_root,
        visualize       = not args.no_visualize,
        save_json       = not args.no_json,
        smooth_window   = args.smooth_window,
    )
    if args.fps:
        cfg.fps = args.fps

    return args, cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args, cfg = parse_args()

    print(f"Dataset : {cfg.dataset_repo_id}")
    print(f"Episodes: {cfg.n_episodes}")
    print(f"Output  : {cfg.output_dir}")
    print()

    # ---- Setup ----
    loader    = EpisodeLoader(cfg)
    segmenter = PrimitiveSegmenter(cfg)
    exporter  = VideoExporter(cfg, cfg.output_dir)

    print("Loading dataset …")
    loader.setup()
    print(f"  FPS: {loader.fps}   |   Episodes available: {loader.num_episodes}")

    export_video = not args.no_video

    # ---- Process episodes ----
    all_results = []
    all_segments = []
    episode_states = []  # keep states for video export

    for ep_idx, states in tqdm(
        loader.iter_episodes(), total=loader.num_episodes, desc="Decomposing"
    ):
        result = segmenter.segment_episode(states, ep_idx)
        all_results.append(result)
        episode_states.append((ep_idx, states))

        # Collect flat segment list for JSON
        for arm in ("left", "right"):
            for seg in result[arm]["segments"]:
                all_segments.append(seg.to_dict())

        # Per-episode PNG
        if cfg.visualize:
            path = visualize_episode(result, ep_idx, loader.fps, cfg.output_dir)
            tqdm.write(f"  Saved figure → {path}")
            ee_path = visualize_ee_pose(result, ep_idx, states, cfg, loader.fps, cfg.output_dir)
            tqdm.write(f"  Saved EE pose → {ee_path}")

    # ---- Video export ----
    if export_video:
        print("\nRendering videos …")
        for (ep_idx, states), result in zip(episode_states, all_results):
            paths = exporter.export_episode(result, ep_idx, states)
            print(f"  Ep {ep_idx}: {len(paths)} video(s) → {os.path.dirname(paths[0])}")

    if not all_results:
        print("No episodes processed. Check dataset name / connectivity.")
        sys.exit(1)

    # ---- Summary visualisation ----
    if cfg.visualize:
        summary_path = visualize_summary(all_results, cfg.output_dir)
        if summary_path:
            print(f"\nSummary figure → {summary_path}")

    # ---- JSON output ----
    if cfg.save_json:
        os.makedirs(cfg.output_dir, exist_ok=True)
        json_path = os.path.join(cfg.output_dir, "segments.json")
        with open(json_path, "w") as f:
            json.dump(all_segments, f, indent=2)
        print(f"Segments JSON  → {json_path}")

    # ---- Console summary ----
    _print_summary(all_segments)


def _print_summary(segments: list) -> None:
    print("\n── Primitive Skill Statistics ────────────────────────────────────")
    counts  : dict[str, int]        = defaultdict(int)
    dur_sum : dict[str, float]      = defaultdict(float)

    for s in segments:
        key = f"{s['arm']:5s}  {s['primitive']:8s}"
        counts[key]   += 1
        dur_sum[key]  += s["duration_seconds"]

    print(f"  {'arm':<5}  {'primitive':<10}  {'count':>6}  {'avg_dur_s':>10}")
    print("  " + "-" * 38)
    for key in sorted(counts):
        arm, prim = key.split()
        n    = counts[key]
        avg  = dur_sum[key] / n
        print(f"  {arm:<5}  {prim:<10}  {n:>6}  {avg:>10.3f}")

    print(f"\n  Total segments: {len(segments)}")
    print("─" * 60)


if __name__ == "__main__":
    main()
