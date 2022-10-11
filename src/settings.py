"""
Constants to control the behavior of the networks and the training are found here.
"""


#: Standard deviation for the layers with nornal initialization of parameters.
STD = 0.07
#: Number of channels of the tensors representing images inside a network.
BASE_DIM = 512

#: Flag that enables a printed summary of the networks before starting training.
SHOW_NETWORK_SUMMARY = False
#: Print batch information every x iterations.
PRINT_BATCH_EVERY_X_ITERATIONS = 20
#: Number of alpha steps per resolution.
ALPHA_STEPS = 1000
#: Weight decay for the optimizers.
WEIGHT_DECAY = 0
#: Number of image samples per resolution.
N_SAMPLES_RES = 600000
#: Garbage collection frequency.
COLLECT_GARBAGE_EVERY_X_ITERATIONS = 50
#: Write to tensorboard frequency.
WRITE_BATCH_EVERY_X_ITERATIONS = PRINT_BATCH_EVERY_X_ITERATIONS

#: Learning rate.
LRATE = 0.00015
#: Betas for the Adam optimizers.
BETAS = (0, 0.99)
#: Epsilon.
EPS = 1e-7

#: Directory where checkpoints reside.
CHECKPOINTS_PATH = 'checkpoints'
#: Directory of the datasets.
DATASETS_DIR = 'datasets'
#: Directory of the tensorboard summaries.
SUMMARIES_DIR = 'summaries'
#: Indicates wheter or not to initialize the traininig from a checkpoint.
START_FROM_CHECKPOINT = False
#: Name of the checkpoint to initialize the training from
#: (name does not include the file extension).
#: Does not have an effect if `START_FROM_CHECKPOINT` is not set.
CHECKPOINT_NAME = None  # not including the extension .pth

#: Intensity of the gradient penalty for the wloss.
GRADIENT_PENALTY_LAMBDA = 3
#: Learning rate of styleGAN.
STGAN_LRATE = LRATE
#: Learning rate of the mapping network.
MAPPN_NETWORK_LR = LRATE * 0.4
#: Learning rate of the discrimiantor.
DISC_LRATE = LRATE

#: The image resolutions the networks are going to be trained with.
#: This variable does not affect the layers/blocks of the networks, only determines
#: the resolutions for the training. To change the resolution accepted by the blocks
#: it is necessary to change the code of the models.
RESOLUTIONS = [8, 16, 32, 64, 128, 256, 512]

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
