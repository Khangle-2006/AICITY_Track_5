import torch
import torchvision.transforms as T

def get_train_transforms(size=(288, 512)):
    """
    Returns a composition of transforms for training.
    Defaults to 512x288 (WTS baseline).
    """
    return T.Compose([
        T.Resize(size, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_val_transforms(size=(288, 512)):
    """
    Returns a composition of transforms for validation/inference.
    """
    return T.Compose([
        T.Resize(size, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def denormalize_image(tensor):
    """
    Converts a normalized tensor [-1, 1] back to [0, 1] range.
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)
