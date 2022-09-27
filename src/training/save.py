import torch
from src.settings import CHECKPOINTS_PATH


def save_checkpoint(
    stgan,
    disc,
    g_optim,
    d_optim,
    res,
    run_id,
    count_batch_global,
    alpha_steps_completed=0,
    path=CHECKPOINTS_PATH
):
    uri = '{}/{}-{}.pth'.format(path, run_id, str(count_batch_global))
    print('Saving checkpoint at: {}'.format(uri))
    try:
        torch.save({
            'stgan_state_dict': stgan.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'g_optim_state_dict': g_optim.state_dict(),
            'd_optim_state_dict': d_optim.state_dict(),
            'resolution': res,
            'global_step': count_batch_global,
            'alpha_steps_completed': alpha_steps_completed,
            'run_id': run_id,
        }, uri)
    except Exception as e:
        print('Saving checkpoint failed with error: {}'.format(e))
