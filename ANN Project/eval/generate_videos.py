"""
Convert GIF evaluation outputs into per-episode MP4s and combined MP4 per model.
Run from project root with the venv activated:

    .\.venv\Scripts\Activate.ps1
    python eval\generate_videos.py

"""
from pathlib import Path
import imageio

PROJECT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = PROJECT / "videos"
OUT_DIR = PROJECT / "videos_mp4"
OUT_DIR.mkdir(exist_ok=True)

# find gifs
gifs = sorted(VIDEOS_DIR.glob("*.gif"))
if not gifs:
    print("No GIFs found in videos/ — nothing to convert.")
    raise SystemExit(0)

# group by prefix before _episode_
from collections import defaultdict
groups = defaultdict(list)
for g in gifs:
    name = g.stem
    if "_episode_" in name:
        prefix = name.split("_episode_")[0]
    else:
        prefix = name
    groups[prefix].append(g)

# Convert each gif to mp4 and also combine per-model
for prefix, files in groups.items():
    print(f"Processing model: {prefix}, {len(files)} episode(s)")
    combined_frames = []
    for g in sorted(files):
        print(" - converting", g.name)
        try:
            # read gif frames
            frames = imageio.mimread(str(g))
            # write per-episode mp4
            ep_name = OUT_DIR / (g.stem + ".mp4")
            imageio.mimsave(str(ep_name), frames, fps=10, macro_block_size=None)
            combined_frames.extend(frames)
        except Exception as e:
            print("   failed to convert", g.name, e)
    # write combined mp4
    if combined_frames:
        out_combined = OUT_DIR / (prefix + "_combined.mp4")
        print(" - writing combined", out_combined.name)
        imageio.mimsave(str(out_combined), combined_frames, fps=10, macro_block_size=None)

print("Done. MP4s saved to:", OUT_DIR)
