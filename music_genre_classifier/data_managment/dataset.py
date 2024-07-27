import os
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader, random_split

DATASET_PATH = "dataset/genres_mel_npz"
MINIBATCH_SIZE = 64
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
    
    def __init__(self, dataset_path: str = DATASET_PATH):
        # Sorting npz files
        npz_files = natsorted(os.listdir(dataset_path))
        # Initialize attributes
        mel_specs = []
        genre_gt = []
        # Iterate through npz files
        for gt_i, genre_i_npz in enumerate(npz_files):
            genre_data = np.load(os.path.join(dataset_path, genre_i_npz))
            # Iterate through mel specs in genre_i
            for mel_spec_i in genre_data["array"]:
                # Creating tensors from np.ndarrays
                mel_spec_tensor = torch.from_numpy(mel_spec_i).float()
                gt_i_tensor = torch.tensor(gt_i)
                # Accumulating mel specs and gt 
                mel_specs.append(mel_spec_tensor)
                genre_gt.append(gt_i_tensor)
        # Go from list of tensors to one tensor
        self.mel_specs = torch.stack(mel_specs)
        self.genre_gt = torch.stack(genre_gt)


    def __getitem__(self, index):
        return self.mel_specs[index], self.genre_gt[index]
    
    def __len__(self):
        return len(self.genre_gt)


music_genre_dataset = MusicGenreDataset()

# Split dataset into train, validation and test set
train_set, valid_set, test_set = random_split(
    dataset=music_genre_dataset,
    lengths=[TRAIN_SET_PCT,VALID_SET_PCT,TEST_SET_PCT],
    generator=torch.Generator().manual_seed(42))

# Dataloaders
train_loader = DataLoader(dataset=train_set,
                        batch_size=MINIBATCH_SIZE,
                        shuffle=True,
                        num_workers=4)

valid_loader = DataLoader(dataset=valid_set,
                        batch_size=MINIBATCH_SIZE,
                        shuffle=True,
                        num_workers=4)

test_loader = DataLoader(dataset=test_set,
                        batch_size=MINIBATCH_SIZE,
                        shuffle=True,
                        num_workers=4)
