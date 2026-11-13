import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

from config import BATCH_SIZE, DATASET_INFO, DIRICHLET_ALPHA, NUM_CLIENTS, REF_FRACTION


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def dirichlet_partition(targets, pool_indices, rng):
    client_indices = [[] for _ in range(NUM_CLIENTS)]
    for label in range(10):
        indices = pool_indices[targets[pool_indices] == label].copy()
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(NUM_CLIENTS, DIRICHLET_ALPHA))
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        for client, part in enumerate(np.split(indices, cuts)):
            client_indices[client].extend(part.tolist())
    return [np.asarray(indices, dtype=np.int64) for indices in client_indices]


def get_data(name, seed, root="./data"):
    shape, mean, std, dataset_type = DATASET_INFO[name]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = dataset_type(root=root, train=True, download=True, transform=transform)
    test = dataset_type(root=root, train=False, download=True, transform=transform)
    targets = np.asarray(train.targets)
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(train))
    ref_indices = rng.choice(all_indices, size=int(len(train) * REF_FRACTION), replace=False)
    private_pool = np.setdiff1d(all_indices, ref_indices, assume_unique=False)

    while True:
        client_indices = dirichlet_partition(targets, private_pool, rng)
        if min(map(len, client_indices)) >= 10:
            break
    ref_x = torch.stack([train[int(i)][0] for i in ref_indices])
    ref_y = torch.tensor([int(targets[i]) for i in ref_indices], dtype=torch.long)
    ref_dataset = TensorDataset(ref_x, ref_y)
    private_sets = [Subset(train, indices.tolist()) for indices in client_indices]
    return train, test, ref_dataset, private_sets, shape
