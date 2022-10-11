"""
This module is an entry point for the training of the network.
To begin the training simply run (from the root directory):

``python -m src.training.train``

The dataset is loaded in this module and is fed to the `TrainingLoop` class to begin
the training.

"""

from torchvision.datasets import CIFAR100
from src.training.training_loop import TrainingLoop
from src.training.setup import get_init_training_vars, get_training_instances, get_checkpoint
from src.settings import START_FROM_CHECKPOINT, DATASETS_DIR


if START_FROM_CHECKPOINT:
    checkpoint = get_checkpoint()
else:
    checkpoint = None
init_training_vars = get_init_training_vars(checkpoint)
stgan, disc, g_optim, d_optim = get_training_instances(checkpoint)
dataset = CIFAR100(DATASETS_DIR, download=True)
training_loop = TrainingLoop(init_training_vars, stgan, disc, g_optim, d_optim, dataset)
# training_loop.start()  # start training
