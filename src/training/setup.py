import datetime
import torch
from torch.optim import Adam
from src.settings import DISC_LRATE, BETAS, EPS
from src.settings import WEIGHT_DECAY, STGAN_LRATE, MAPPN_NETWORK_LR, CHECKPOINTS_PATH
from src.settings import ALPHA_STEPS, CHECKPOINT_NAME
from src.models.discriminator import Discriminator
from src.models.generator.networks import StyleGAN
from src.device import device


def get_checkpoint():
    checkpoint_uri = '{}/{}.pth'.format(CHECKPOINTS_PATH, CHECKPOINT_NAME)
    try:
        checkpoint = torch.load(checkpoint_uri)
    except FileNotFoundError:
        print('Can not find checkpoint: {}'.format(CHECKPOINT_NAME))
    return checkpoint


def get_init_training_vars(checkpoint):
    if checkpoint:
        print('Loading from: {}/{}'.format(CHECKPOINTS_PATH, CHECKPOINT_NAME))
        start_res = checkpoint['resolution']
        count_batch = checkpoint['global_step'] + 1
        alpha_steps_completed = checkpoint['alpha_steps_completed']
        run_id = checkpoint['run_id']
        log_filename_suffix = str(count_batch)
    else:
        print('Initializing training data with default values.')
        start_res = 8
        count_batch = 0
        alpha_steps_completed = 0
        run_id = str(datetime.datetime.now().strftime("%d_%m-%H:%M"))
        log_filename_suffix = ''
    if ALPHA_STEPS > 1:
        start_alpha = (alpha_steps_completed) * (1 / (ALPHA_STEPS - 1))
        # print('start alpha: {}'.format(start_alpha))
    else:
        start_alpha = 1
    if start_alpha == 0:
        start_alpha = 0.001

    init_training_vars = {
        'start_res': start_res,
        'count_batch': count_batch,
        'alpha_steps_completed': alpha_steps_completed,
        'run_id': run_id,
        'log_filename_suffix': log_filename_suffix,
    }
    return init_training_vars


def load_state_dicts(stgan, disc, g_optim, d_optim, checkpoint):
    stgan.load_state_dict(checkpoint['stgan_state_dict'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    g_optim.load_state_dict(checkpoint['g_optim_state_dict'])
    d_optim.load_state_dict(checkpoint['d_optim_state_dict'])


def get_training_instances(checkpoint):
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
