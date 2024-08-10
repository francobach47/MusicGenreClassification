import os
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader, random_split

# Dataset split
TRAIN_SET_PCT = 0.8
VALID_SET_PCT = 0.15
TEST_SET_PCT = 0.05

genres_map = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "Disco",
    4: "HipHop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock",
}


# Dataset class
class MusicGenreDataset(Dataset):

    def __init__(self, dataset_path: str):

        dataset_path = get_data_path(dataset_path)
        mel_specs = []
        genre_gt = []

        for genre_dir in os.listdir(dataset_path):
            genre_path = os.path.join(dataset_path, genre_dir)
            npz_files = natsorted(os.listdir(genre_path))
            for gt_i, genre_i_npz in enumerate(npz_files):
                genre_data = np.load(os.path.join(genre_path, genre_i_npz))
                for mel_spec_i in genre_data["spectrogram"]:
                    mel_spec_tensor = torch.from_numpy(mel_spec_i).float()
                    gt_i_tensor = torch.tensor(gt_i)
                    mel_specs.append(mel_spec_tensor)
                    genre_gt.append(gt_i_tensor)
        self.mel_specs = torch.stack(mel_specs)
        self.genre_gt = torch.stack(genre_gt)

    def __getitem__(self, index):
        return self.mel_specs[index], self.genre_gt[index]

    def __len__(self):
        return len(self.genre_gt)


def get_train_loader(dataset_path: str, batch_size: int = 64) -> DataLoader:
    dataset = MusicGenreDataset(dataset_path)
    train_set, _, _ = random_split(
        dataset=dataset,
        lengths=[TRAIN_SET_PCT, VALID_SET_PCT, TEST_SET_PCT],
        generator=torch.Generator().manual_seed(42),
    )
    return DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )


def get_valid_loader(dataset_path: str, batch_size: int = 64) -> DataLoader:
    dataset = MusicGenreDataset(dataset_path)
    _, valid_set, _ = random_split(
        dataset=dataset,
        lengths=[TRAIN_SET_PCT, VALID_SET_PCT, TEST_SET_PCT],
        generator=torch.Generator().manual_seed(42),
    )
    return DataLoader(
        dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=4
    )


def get_test_loader(dataset_path: str, batch_size: int = 64) -> DataLoader:
    dataset = MusicGenreDataset(dataset_path)
    _, _, test_set = random_split(
        dataset=dataset,
        lengths=[TRAIN_SET_PCT, VALID_SET_PCT, TEST_SET_PCT],
        generator=torch.Generator().manual_seed(42),
    )
    return DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )


def get_data_path(base_path):
    return os.path.join(base_path, "genres_mel_npz")
