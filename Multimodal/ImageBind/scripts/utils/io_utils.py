import json
import numpy as np
import os

def save_features(results, full_transcript, output_dir, video_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save MetaData
    meta_path = os.path.join(output_dir, f"{video_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({"transcript": full_transcript, "segments": len(results)}, f)

    # Save Embeddings (The heavy part)
    emb_path = os.path.join(output_dir, f"{video_name}_embeddings.npz")
    np.savez_compressed(
        emb_path,
        vision=np.array([r['vision_emb'] for r in results]),
        audio=np.array([r['audio_emb'] for r in results]),
        text=np.array([r['text_emb'] for r in results]),
        times=np.array([(r['start'], r['end']) for r in results])
    )
    print(f"✓ Saved features to {output_dir}")
    

def load_features(emb_path):
    """Loads the NPZ file back into a dictionary."""
    return np.load(emb_path, allow_pickle=True)