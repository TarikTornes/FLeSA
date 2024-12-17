import torch

def get_device():
    """
    This function searches for the suitable device on which
    the Machine Learning tasks will be performed on.

    Return:
        device (str): the device that will be used
                        -> mps, cuda or cpu
    """


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

