from torchvision import datasets


NUM_CLIENTS = 10
DIRICHLET_ALPHA = 0.1
REF_FRACTION = 0.05
LR = 3e-4
BATCH_SIZE = 32
TEMPERATURE = 5.0
ROUNDS = 5
DISTILL_EPOCHS = 20
PHASE_MAX_EPOCHS = 100
PHASE_PATIENCE = 3
PHASE_MIN_DELTA = 1e-3
SEEDS = [0, 1, 2, 3, 4]
THRESHOLDS = {"mnist": 97.0, "fmnist": 80.0, "cifar10": 63.0}
DATASET_INFO = {
    "mnist": ((1, 28, 28), (0.1307,), (0.3081,), datasets.MNIST),
    "fmnist": ((1, 28, 28), (0.2860,), (0.3530,), datasets.FashionMNIST),
    "cifar10": ((3, 32, 32), (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), datasets.CIFAR10),
}
