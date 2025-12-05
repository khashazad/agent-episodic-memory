import os
import torch
import torch.nn as nn
from pathlib import Path

def build_mlp(input_dim, output_dim, hidden_dim, num_layers=2, add_input_activation=False, **kwargs):
    """Build a simple MLP."""
    layers = []
    
    if add_input_activation:
        layers.append(nn.ReLU())
    
    for i in range(num_layers):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dim))
        elif i == num_layers - 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        if i < num_layers - 1:
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)

def get_device(model):
    """Get device of a model."""
    return next(model.parameters()).device

def basic_image_tensor_preprocess(frames, mean=(0.3331, 0.3245, 0.3051), std=(0.2439, 0.2493, 0.2873)):
    """Basic image preprocessing."""
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    
    # Handle different tensor shapes
    if frames.ndim == 4:  # [B, C, H, W]
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(frames.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(frames.device)
    elif frames.ndim == 5:  # [B, T, C, H, W]
        mean = torch.tensor(mean).view(1, 1, 3, 1, 1).to(frames.device)
        std = torch.tensor(std).view(1, 1, 3, 1, 1).to(frames.device)
    else:
        raise ValueError(f"Unsupported tensor shape: {frames.shape}")
    
    frames = (frames - mean) / std
    return frames

def f_expand(path):
    """Expand file path."""
    return Path(path).expanduser().resolve()

def f_exists(path):
    """Check if file exists."""
    return Path(path).exists()

def torch_load(path):
    """Load torch checkpoint."""
    return torch.load(path, map_location='cpu')

def load_state_dict(model, state_dict, strip_prefix=None, strict=True):
    """Load state dict into model."""
    if strip_prefix:
        # Remove prefix from keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(strip_prefix):
                new_key = key[len(strip_prefix):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    return model.load_state_dict(state_dict, strict=strict)