#!/usr/bin/env python3
"""Convert V-JEPA-style UR5 traj datasets to diffusion-pipe flat layout.

Source layout (per task dir):
    <src>/traj_NNNNNN/recordings/MP4/left.mp4

Output layout:
    <out>/train/<task>/traj_NNNNNN.mp4   (symlink)
    <out>/train/<task>/traj_NNNNNN.txt   (caption)
    <out>/val/<task>/...

Trajs sorted by name; every 10th (i % 10 == 9) goes to val.
"""
from pathlib import Path

DATASETS = [
    ("/home/riftuser/datasets/ur5_vjepa_0501_bowl",    "bowl",    "pick up the blue bowl and place it into the black bowl"),
    ("/home/riftuser/datasets/ur5_vjepa_0519_cup",     "cup",     "pick up the cup and place it into the black bowl"),
    ("/home/riftuser/datasets/ur5_vjepa_0519_penCase", "penCase", "pick up the pencil case and place it into the brown box"),
    ("/home/riftuser/datasets/ur5_vjepa_0521_bowlBox", "bowlBox", "pick up the bowl and place it into the brown box"),
]
OUTPUT_ROOT = Path("/home/riftuser/datasets/diffusion_pipe")
VAL_EVERY = 10


def link_one(src_mp4: Path, dst_dir: Path, name: str, caption: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    link = dst_dir / f"{name}.mp4"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src_mp4.resolve())
    (dst_dir / f"{name}.txt").write_text(caption + "\n")


def main() -> None:
    for src_str, task, caption in DATASETS:
        src = Path(src_str)
        trajs = sorted(p for p in src.glob("traj_*") if p.is_dir())
        n_train = n_val = n_skip = 0
        for i, traj in enumerate(trajs):
            mp4 = traj / "recordings" / "MP4" / "left.mp4"
            if not mp4.exists():
                print(f"[{task}] skip missing: {mp4}")
                n_skip += 1
                continue
            split = "val" if i % VAL_EVERY == VAL_EVERY - 1 else "train"
            link_one(mp4, OUTPUT_ROOT / split / task, traj.name, caption)
            if split == "val":
                n_val += 1
            else:
                n_train += 1
        print(f"[{task}] train={n_train}  val={n_val}  skipped={n_skip}")


if __name__ == "__main__":
    main()
