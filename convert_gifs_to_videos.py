"""
Convert GIF episodes to MP4 videos per model
Creates one video per model combining episodes 1-10
"""

from pathlib import Path
from PIL import Image
import imageio
import numpy as np

# Configuration
VIDEO_DIR = Path("ANN Project/videos")
OUTPUT_DIR = VIDEO_DIR / "compiled"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model patterns
MODELS = {
    'DQN': 'DQN_highway_10k',
    'Double_DQN': 'double_dqn_highway',
    'Dueling_DQN': 'dueling_double_dqn_highway'
}

def gif_to_frames(gif_path):
    """Extract frames from GIF"""
    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = img.copy()
            frames.append(np.array(frame.convert('RGB')))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames

def create_model_video(model_name, pattern, episodes=10):
    """Combine episodes 1-10 into single video for a model"""
    print(f"\nProcessing {model_name}...")
    
    all_frames = []
    
    for ep in range(1, episodes + 1):
        gif_file = VIDEO_DIR / f"{pattern}_episode_{ep}.gif"
        
        if not gif_file.exists():
            print(f"  ⚠ Missing: episode {ep}")
            continue
        
        print(f"  Loading episode {ep}...")
        frames = gif_to_frames(gif_file)
        
        # Add title frame
        title_frame = frames[0].copy()
        # Simple text overlay (white background bar at top)
        title_frame[0:30, :] = 255
        
        all_frames.append(title_frame)
        all_frames.extend(frames)
        
        # Add blank frames between episodes
        blank = np.ones_like(frames[0]) * 128
        all_frames.extend([blank] * 10)
    
    if not all_frames:
        print(f"  ✗ No frames found for {model_name}")
        return
    
    # Save as MP4
    output_file = OUTPUT_DIR / f"{model_name}_episodes_1-10.mp4"
    print(f"  Saving video: {output_file.name}")
    
    imageio.mimsave(
        output_file,
        all_frames,
        fps=10,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    )
    
    print(f"  ✓ Saved: {len(all_frames)} frames, {output_file}")

def main():
    print("="*70)
    print("CONVERTING GIF EPISODES TO MP4 VIDEOS")
    print("="*70)
    
    for model_name, pattern in MODELS.items():
        try:
            create_model_video(model_name, pattern)
        except Exception as e:
            print(f"  ✗ Error processing {model_name}: {e}")
    
    print("\n" + "="*70)
    print("✓ VIDEO COMPILATION COMPLETE")
    print("="*70)
    print(f"\nVideos saved in: {OUTPUT_DIR}")
    print("\nFiles created:")
    for model in MODELS.keys():
        video_file = OUTPUT_DIR / f"{model}_episodes_1-10.mp4"
        if video_file.exists():
            size_mb = video_file.stat().st_size / (1024*1024)
            print(f"  ✓ {video_file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
