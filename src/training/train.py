"""
This module is an entry point for the training of the network.
To begin the training simply run (from the root directory):

``python src/training/train.py``

The dataset is loaded in this module and is fed to the `TrainingLoop` class to begin
the training.

"""

if __name__ == '__main__':
    # Appending src package to sys.path.
    import sys
    import os
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)  # src/
    sys.path.insert(0, parentdir)

    # Training
    from training.training_loop import TrainingLoop
    from training.setup import get_init_training_vars, get_training_instances, get_checkpoint
    from settings import START_FROM_CHECKPOINT, DATASETS_DIR, DATASET

    if START_FROM_CHECKPOINT:
        checkpoint = get_checkpoint()
    else:
        checkpoint = None
    init_training_vars = get_init_training_vars(checkpoint)
    stgan, disc, g_optim, d_optim = get_training_instances(checkpoint)
    dataset = DATASET(DATASETS_DIR, download=True)
    training_loop = TrainingLoop(init_training_vars, stgan, disc, g_optim, d_optim, dataset)

    training_loop.start()  # start training
