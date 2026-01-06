from pathlib import Path
import imageio

PROJECT = Path(__file__).resolve().parents[1]
IN = PROJECT / "videos_mp4" / "dueling_retry.mp4"
OUT = PROJECT / "videos_mp4" / "dueling_slow.mp4"

if not IN.exists():
    print("Input file not found:", IN)
    raise SystemExit(1)

reader = imageio.get_reader(str(IN))
frames = []
for frame in reader:
    frames.append(frame)
reader.close()

# write with half fps (if original was 10, new will be 5)
imageio.mimsave(str(OUT), frames, fps=5)
print("Wrote", OUT)
