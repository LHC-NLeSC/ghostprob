from torch.utils.data import Dataset

label = "ghost"

training_columns_forward = [
    "best_pt",
    "chi2",
    "first_qop",
    "n_scifi",
    "n_velo",
    "ndof",
    "pr_forward_quality",
    "qop",
]

training_columns_matching = [
    "best_pt",
    "chi2",
    "first_qop",
    "n_scifi",
    "n_velo",
    "ndof",
    "pr_match_chi2",
    "pr_match_dtx",
    "pr_match_dty",
    "pr_match_dx",
    "pr_match_dy",
    "pr_seed_chi2",
    "qop",
]

boundaries = {"best_pt": (0, 1e4), "chi2": (0, 400)}


class GhostDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
