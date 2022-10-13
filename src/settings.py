"""
Constants to control the behavior of the networks and the training are found here.
"""

# importing dataset
from torchvision.datasets import FashionMNIST
#: :meta hide-value:
#:
#: Image dataset used to train the networks.
DATASET = FashionMNIST

#: :meta hide-value:
#:
#: Number of color channels in the images of `DATASET`. Usually 3 channels for
#: color images, 1 for black and white images.
DATASET_CHANNELS = 1

#: :meta hide-value:
#:
#: Standard deviation for the layers with nornal initialization of parameters.
STD = 0.07

#: :meta hide-value:
#:
#: Number of channels of the tensors representing images inside a network.
BASE_DIM = 512

#: :meta hide-value:
#:
#: Flag that enables a printed summary of the networks before starting training.
SHOW_NETWORK_SUMMARY = False

#: :meta hide-value:
#:
#: Print batch information every x iterations.
PRINT_BATCH_EVERY_X_ITERATIONS = 20

#: :meta hide-value:
#:
#: Number of alpha steps per resolution.
ALPHA_STEPS = 1000

#: :meta hide-value:
#:
#: Weight decay for the optimizers.
WEIGHT_DECAY = 0

#: :meta hide-value:
#:
#: Number of image samples per resolution.
N_SAMPLES_RES = 600000

#: :meta hide-value:
#:
#: Garbage collection frequency.
COLLECT_GARBAGE_EVERY_X_ITERATIONS = 50

#: :meta hide-value:
#:
#: Write to tensorboard frequency.
WRITE_BATCH_EVERY_X_ITERATIONS = PRINT_BATCH_EVERY_X_ITERATIONS

#: :meta hide-value:
#:
#: Learning rate.
LRATE = 0.00015

#: :meta hide-value:
#:
#: Betas for the Adam optimizers.
BETAS = (0, 0.99)

#: :meta hide-value:
#:
#: Epsilon.
EPS = 1e-7

#: :meta hide-value:
#:
#: Directory where checkpoints reside.
CHECKPOINTS_PATH = 'checkpoints'

#: :meta hide-value:
#:
#: Directory of the datasets.
DATASETS_DIR = 'datasets'

#: :meta hide-value:
#:
#: Directory of the tensorboard summaries.
SUMMARIES_DIR = 'summaries'

#: :meta hide-value:
#:
#: Indicates wheter or not to initialize the traininig from a checkpoint.
START_FROM_CHECKPOINT = False

#: :meta hide-value:
#:
#: Name of the checkpoint to initialize the training from
#: (name does not include the file extension).
#: Does not have an effect if `START_FROM_CHECKPOINT` is not set.
CHECKPOINT_NAME = None  # not including the extension .pth

#: :meta hide-value:
#:
#: Intensity of the gradient penalty for the wloss.
GRADIENT_PENALTY_LAMBDA = 3

#: :meta hide-value:
#:
#: Learning rate of styleGAN.
STGAN_LRATE = LRATE

#: :meta hide-value:
#:
#: Learning rate of the mapping network.
MAPPN_NETWORK_LR = LRATE * 0.4

#: :meta hide-value:
#:
#: Learning rate of the discrimiantor.
DISC_LRATE = LRATE

#: :meta hide-value:
#:
#: The image resolutions the networks are going to be trained with.
#: This constant does not affect the layers/blocks of the networks, only determines
#: the resolutions for the training. To change the resolution accepted by the blocks
#: it is necessary to change the code of the models.
#: If you modify this constant, you also need to modify `SAVE_EVERY_X_ITERATIONS` and
#: `RES_BATCH_SIZE`.
RESOLUTIONS = [8, 16, 32, 64, 128, 256, 512]

#: :meta hide-value:
#:
#: Save a checkpoint every x iteration depending on what resolution the training is at.
SAVE_EVERY_X_ITERATIONS = {
    '8': 8000,
    '16': 8000,
    '32': 8000,
    '64': 5000,
    '128': 2000,
    '256': 1250,
    '512': 1250,
    '1024': 1024
}

#: :meta hide-value:
#:
#: Size of the training batches depending on what resolution the training is at.
RES_BATCH_SIZE = {
    '4': 16,
    '8': 16,
    '16': 16,
    '32': 16,
    '64': 16,
    '128': 16,
    '256': 14,
    '512': 6,
    '1024': 3
}
