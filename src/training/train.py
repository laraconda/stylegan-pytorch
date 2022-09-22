from torchvision.datasets import CIFAR100
from src.training.training_loop import TrainingLoop
from src.training.setup import get_init_training_vars, get_training_instances, get_checkpoint
# from training.training_loop import TrainingLoop
from src.settings import START_FROM_CHECKPOINT


if START_FROM_CHECKPOINT:
    checkpoint = get_checkpoint()
else:
    checkpoint = None
init_training_vars = get_init_training_vars(checkpoint)
stgan, disc, g_optim, d_optim = get_training_instances(checkpoint)
dataset = CIFAR100('datasets/', download=True)
print(f'dataset type: {type(dataset)}')
training_loop = TrainingLoop(init_training_vars, stgan, disc, g_optim, d_optim, dataset)
training_loop.run()
