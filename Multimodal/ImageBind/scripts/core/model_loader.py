"""
ImageBind Model Loader
Handles initialization of ImageBind and Whisper models.
"""
import torch
import whisper
from imagebind.models import imagebind_model


def load_imagebind(device="cuda:0", verbose=True):
    """
    Load ImageBind-huge model.
    
    Args:
        device: Device to load model on ('cuda:0' or 'cpu')
        verbose: Print loading messages
    
    Returns:
        model: ImageBind model in eval mode
    """
    if verbose:
        print("Loading ImageBind-huge model...")
        print("This downloads ~1.2GB on first run (2-3 minutes)...\n")
    
    try:
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(device)
        
        if verbose:
            print("✓ ImageBind model loaded successfully!")
            print(f"Device: {device}")
            
            if torch.cuda.is_available() and device.startswith('cuda'):
                mem = torch.cuda.memory_allocated(0) / 1e9
                print(f"GPU Memory Used: {mem:.2f} GB")
        
        return model
        
    except Exception as e:
        print(f"✗ Failed to load ImageBind: {e}")
        raise


def load_whisper(model_size="base", verbose=True):
    """
    Load Whisper model for transcription.
    
    Args:
        model_size: Size of Whisper model
                   Options: 'tiny', 'base', 'small', 'medium', 'large'
        verbose: Print loading messages
    
    Returns:
        model: Whisper model
    """
    if verbose:
        print(f"Loading Whisper model (size: {model_size})...")
    
    try:
        model = whisper.load_model(model_size)
        
        if verbose:
            print(f"✓ Whisper-{model_size} loaded successfully!")
        
        return model
        
    except Exception as e:
        print(f"✗ Failed to load Whisper: {e}")
        raise


def get_device():
    """
    Get best available device.
    
    Returns:
        device: 'cuda:0' if GPU available, else 'cpu'
    """
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print(" No GPU available, using CPU (will be slower)")
    
    return device


def load_models(imagebind_device=None, whisper_size="base", verbose=True):
    """
    Load both ImageBind and Whisper models.
    
    Args:
        imagebind_device: Device for ImageBind (auto-detect if None)
        whisper_size: Whisper model size
        verbose: Print loading messages
    
    Returns:
        dict: {'imagebind': model, 'whisper': model, 'device': device}
    """
    if verbose:
        print("=" * 60)
        print("LOADING MODELS")
        print("=" * 60 + "\n")
    
    # Get device
    if imagebind_device is None:
        device = get_device()
    else:
        device = imagebind_device
    
    # Load models
    imagebind = load_imagebind(device, verbose)
    whisper_model = load_whisper(whisper_size, verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ All models loaded successfully!")
        print("=" * 60)
    
    return {
        'imagebind': imagebind,
        'whisper': whisper_model,
        'device': device
    }


# Convenience function for quick setup
def quick_load(whisper_size="base"):
    """
    Quick model loading with sensible defaults.
    
    Returns:
        imagebind_model, whisper_model, device
    """
    models = load_models(whisper_size=whisper_size, verbose=True)
    return models['imagebind'], models['whisper'], models['device']