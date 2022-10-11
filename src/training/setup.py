"""
Declaration of functions that help set up a training session.
"""

from sys import stderr
from datetime import datetime
import torch
from torch.optim import Adam
from src.settings import DISC_LRATE, BETAS, EPS
from src.settings import WEIGHT_DECAY, STGAN_LRATE, MAPPN_NETWORK_LR, CHECKPOINTS_PATH
from src.settings import ALPHA_STEPS, CHECKPOINT_NAME
from src.models.discriminator import Discriminator
from src.models.generator.networks import StyleGAN
from src.device import device


def get_checkpoint():
    """
    Based on the checkpoints path and the checkpoint name specified in settings
    (constants CHECKPOINTS_PATH, CHECKPOINT_NAME), tries to load the checkpoint.
    This function assumes the checkpoint file has a 'pth' extension.

    If the file is not found, an error message written to stderr will be printed
    and None will be returned

    Returns
    -------
    dict
        The loaded checkpoint.
    None
        If the checkpoint is not found at the specified location.
    """
    checkpoint_uri = '{}/{}.pth'.format(CHECKPOINTS_PATH, CHECKPOINT_NAME)
    try:
        checkpoint = torch.load(checkpoint_uri)
    except FileNotFoundError:
        print('Can not find checkpoint: {}'.format(CHECKPOINT_NAME), file=stderr)
    return checkpoint


def get_init_training_vars(checkpoint=None):
    """
    Collects the necesary variables to begin or continue a training session.

    If a checkpoint is provided, then the variables are going to be extracted from it.
    Otherwise the variables are initialized with the intention of beginning a completely
    new training.

    Parameters
    ----------
    checkpoint: dict, optional
        Contains information from a previously halted training session (default is None).

    Returns
    -------
    dict
        The necessary variables to initialize a training session.
    """
    if checkpoint:
        print('Loading training from: {}/{}.pth'.format(CHECKPOINTS_PATH, CHECKPOINT_NAME))
        start_res = checkpoint['resolution']
        global_step = checkpoint['global_step'] + 1
        alpha_steps_completed = checkpoint['alpha_steps_completed']
        run_id = checkpoint['run_id']
        log_filename_suffix = str(global_step)
    else:
        print('Initializing training data with default values (no checkpoint).')
        start_res = 8
        global_step = 0
        alpha_steps_completed = 0
        run_id = str(datetime.now().strftime("%d_%m-%H:%M"))
        log_filename_suffix = ''
    if ALPHA_STEPS > 1:
        start_alpha = (alpha_steps_completed) * (1 / (ALPHA_STEPS - 1))
    else:
        start_alpha = 1
    if start_alpha == 0:
        start_alpha = 0.001

    init_training_vars = {
        'start_res': start_res,
        'global_step': global_step,
        'alpha_steps_completed': alpha_steps_completed,
        'run_id': run_id,
        'log_filename_suffix': log_filename_suffix,  # for summary writter
        'start_alpha': start_alpha
    }
    return init_training_vars


def load_state_dicts(stgan, disc, g_optim, d_optim, checkpoint):
    """
    Loads a state on the networks and their optimizers from a checkpoint dict.

    This function changes the training parameters of each of the networks and optimizers.

    Parameters
    ----------
    stgan: StyleGAN
        An instance of the generator network.
    disc: Discriminator
        Instance of the discriminator.
    g_optim: Optimizer
        Instance of the optimizer of the generator.
    d_optim: Optimizer
        Instance of the optimizer of the discriminator.
    """
    stgan.load_state_dict(checkpoint['stgan_state_dict'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    g_optim.load_state_dict(checkpoint['g_optim_state_dict'])
    d_optim.load_state_dict(checkpoint['d_optim_state_dict'])


def get_training_instances(checkpoint=None):
    """
    Initializes an instance of each of the StyleGAN and Discriminator networks and
    an instance of each of their optimizers.

    If the `checkpoint` parameter is not None, then the trainable parameters of the
    networks and their optimizers will be updated using the checkpoint data.

    Parameters
    ----------
    checkpoint: dict, optional
        Contains information from a previously halted training session (default is None).

    Returns
    -------
    tuple:
        The four initialized instances: StyleGAN and Discriminator and their respecitve
        optimizers.
    """
    stgan = StyleGAN().to(device)
    disc = Discriminator().to(device)
    g_optim = Adam([
        {'params': stgan.synthn.conv_blocks.parameters()},
        {'params': stgan.synthn.to_rgb.parameters()},
        {'params': stgan.mappn.parameters(), 'lr': MAPPN_NETWORK_LR},
    ], lr=STGAN_LRATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
    d_optim = Adam(
        disc.parameters(),
        lr=DISC_LRATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY
    )
    if checkpoint:
        load_state_dicts(stgan, disc, g_optim, d_optim, checkpoint)
    return stgan, disc, g_optim, d_optim
