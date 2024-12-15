import torch

def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
        # print("Using device: mps")
    elif torch.backends.cuda.is_available():
        device = "cuda"
        # print("Using device: cuda")
    else:
        device = "cpu"
        # print("Using device: cpu")

    return device

