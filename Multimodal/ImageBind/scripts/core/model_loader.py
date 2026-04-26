"""
ImageBind Model Loader
Handles initialization of ImageBind, Whisper, BLIP, and Whisper-AT models.
"""
import subprocess
import sys

import torch
import whisper
from imagebind.models import imagebind_model
from transformers import BlipForConditionalGeneration, BlipProcessor


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


def load_blip(
    device="cuda",
    model_name="Salesforce/blip-image-captioning-large",
    verbose=True,
):
    """
    Load BLIP captioning model and processor.

    Args:
        device: Device to load model on ('cuda' or 'cpu')
        model_name: BLIP model id from Hugging Face
        verbose: Print loading messages

    Returns:
        tuple: (processor, blip_model)
    """
    if verbose:
        print(f"Loading BLIP model ({model_name})...")

    try:
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        processor = BlipProcessor.from_pretrained(model_name)
        blip_model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device).eval()

        if verbose:
            print("✓ BLIP model loaded successfully!")

        return processor, blip_model

    except Exception as e:
        print(f"✗ Failed to load BLIP: {e}")
        raise


def _ensure_whisper_at_installed(verbose=True):
    """Install whisper-at if missing."""
    try:
        import whisper_at  # noqa: F401
        return
    except ImportError:
        if verbose:
            print("whisper-at not found. Installing with --no-deps...")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "whisper-at"],
        check=True,
    )


def load_whisper_at(model_size="base", auto_install=True, verbose=True):
    """
    Load Whisper-AT model for ASR + AudioSet tagging.

    Args:
        model_size: Size of Whisper-AT model
        auto_install: Install whisper-at if not already installed
        verbose: Print loading messages

    Returns:
        tuple: (whisper_at_model, parse_at_label_function)
    """
    if verbose:
        print(f"Loading Whisper-AT model (size: {model_size})...")

    try:
        if auto_install:
            _ensure_whisper_at_installed(verbose=verbose)

        import whisper_at as whisper_at_model
        from whisper_at import parse_at_label

        model = whisper_at_model.load_model(model_size)

        if verbose:
            print(f"✓ Whisper-AT-{model_size} loaded successfully!")

        return model, parse_at_label

    except Exception as e:
        print(f"✗ Failed to load Whisper-AT: {e}")
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


def load_models(
    imagebind_device=None,
    whisper_size="base",
    blip_model_name="Salesforce/blip-image-captioning-large",
    whisper_at_size="base",
    auto_install_whisper_at=True,
    verbose=True,
):
    """
    Load ImageBind, Whisper, BLIP, and Whisper-AT models.
    
    Args:
        imagebind_device: Device for ImageBind (auto-detect if None)
        whisper_size: Whisper model size
        blip_model_name: Hugging Face model name for BLIP
        whisper_at_size: Whisper-AT model size
        auto_install_whisper_at: Install whisper-at if missing
        verbose: Print loading messages
    
    Returns:
        dict: loaded models and utility functions
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
    blip_processor, blip_model = load_blip(
        device=device,
        model_name=blip_model_name,
        verbose=verbose,
    )
    whisper_at_model, parse_at_label = load_whisper_at(
        model_size=whisper_at_size,
        auto_install=auto_install_whisper_at,
        verbose=verbose,
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ All models loaded successfully!")
        print("=" * 60)
    
    return {
        'imagebind': imagebind,
        'whisper': whisper_model,
        'blip_processor': blip_processor,
        'blip': blip_model,
        'whisper_at': whisper_at_model,
        'parse_at_label': parse_at_label,
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


def quick_load_all(whisper_size="base"):
    """
    Quick loading for all models used in notebook Cell 4.

    Returns:
        dict with imagebind, whisper, blip_processor, blip, whisper_at, parse_at_label, device
    """
    return load_models(whisper_size=whisper_size, verbose=True)