import torch

def loadData():
    dst_start = []
    dst_end = []
    isFilled = []
    src = []

    with open("data/dst_start.txt", mode="r") as f:
        dst_start = f.readlines()
        dst_start = [int(idx) for idx in dst_start]
    
    with open("data/dst_end.txt", mode="r") as f:
        dst_end = f.readlines()
        dst_end = [int(idx) for idx in dst_end]

    with open("data/isFilled.txt", mode="r") as f:
        isFilled = f.readlines()
        isFilled = [int(idx) for idx in isFilled]

    with open("data/src.txt", mode="r") as f:
        src = f.readlines()

    return torch.tensor(dst_start), torch.tensor(dst_end), torch.tensor(isFilled), src