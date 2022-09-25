#
STD = 0.07
BASE_DIM = 512
#
SHOW_NETWORK_SUMMARY = False
PRINT_BATCH_EVERY_X_ITERATIONS = 300
ALPHA_STEPS = 1000
WEIGHT_DECAY = 0
N_SAMPLES_RES = 600000
COLLECT_GARBAGE_EVERY_X_ITERATIONS = 50

LRATE = 0.00015
BETAS = (0, 0.99)
EPS = 1e-7

CHECKPOINTS_PATH = 'checkpoints'
DATASETS_DIR = 'datasets'
START_FROM_CHECKPOINT = True
CHECKPOINT_NAME = '24_09-18:53-0'  # not including the extension .pth

GRADIENT_PENALTY_LAMBDA = 3
STGAN_LRATE = LRATE
MAPPN_NETWORK_LR = LRATE * 0.4
DISC_LRATE = LRATE

# the image resolutions the networks are going to be trained with
RESOLUTIONS = [8, 16, 32, 64, 128, 256, 512]


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
