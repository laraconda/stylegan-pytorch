from src.training.setup import get_init_training_vars, get_training_instances, get_checkpoint
# from training.training_loop import TrainingLoop
from src.settings import START_FROM_CHECKPOINT


if START_FROM_CHECKPOINT:
    checkpoint = get_checkpoint()
else:
    checkpoint = None
init_training_vars = get_init_training_vars(checkpoint)
stgan, disc, g_optim, d_optim = get_training_instances(checkpoint)

# trainig_loop = TrainingLoop(init_training_vars, stgan, disc, g_optim, d_optim)
