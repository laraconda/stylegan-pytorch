"""
Definition of the function used to save a checkpoint of a training session.
"""

from sys import stderr
import torch
from settings import CHECKPOINTS_PATH


def save_checkpoint(
    stgan_state_dict,
    disc_state_dict,
    g_optim_state_dict,
    d_optim_state_dict,
    res,
    run_id,
    global_step,
    alpha_steps_completed=0,
    path=CHECKPOINTS_PATH
):
    """
    Saves a checkpoint dict on the specified location.

    Prints the location where the file (assuming no errors occur) is going to be saved.

    Parameters
    ----------
    stgan_state_dict: dict
        State dictionary of the generator network.
    disc_state_dict: dict
        State dictionary of the discriminator network.
    g_optim_state_dict: dict
        State dictionary of the generator optimizer.
    d_optim_state_dict: dict
        State dictionary of the discriminator optimizer.
    res: int
        Current resolution of the training.
    run_id: string
        Id of the training.
    global_step: int
        Current global step of the training.
    alpha_steps_completed: int, optional
        Number of alpha steps completed in this resoution (default is 0).
    path: string, optional
        Path where the 'pth' file will be saved
        (default is `CHECKPOINTS_PATH`, defined in settings).

    Notes
    -----
    If the saving fails, an error message will be written to `stderr`.
    """
    uri = '{}/{}-{}.pth'.format(path, run_id, str(global_step))
    print('Saving checkpoint at: {}'.format(uri))
    try:
        torch.save({
            'stgan_state_dict': stgan_state_dict,
            'disc_state_dict': disc_state_dict,
            'g_optim_state_dict': g_optim_state_dict,
            'd_optim_state_dict': d_optim_state_dict,
            'resolution': res,
            'global_step': global_step,
            'alpha_steps_completed': alpha_steps_completed,
            'run_id': run_id,
        }, uri)
    except Exception as e:
        print('Saving checkpoint failed with error: {}'.format(e), file=stderr)
