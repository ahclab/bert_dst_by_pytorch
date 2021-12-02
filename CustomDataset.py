import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, isIncluding,start_idxs, end_idxs):
        self.encodings = encodings
        self.isIncluding = isIncluding
        self.start_idxs = start_idxs
        self.end_idxs = end_idxs

    def __getitem__(self, idx):
        result_dict = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        result_dict["labels"] = self.isIncluding[idx]
        result_dict["start_idxs"] = self.start_idxs[idx]
        result_dict["end_idxs"] = self.end_idxs[idx]
        return result_dict

    def __len__(self):
        return len(self.encodings.input_ids)