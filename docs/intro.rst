Intro
=====

Welcome to the documentation of StyleGAN PyTorch!

This is an implementation focused on the *training* of the StyleGAN architecture using PyTorch.

It is important to note that this project only works with square images.

Train
-----
Before starting the training process, make sure the settings (found in src/settings.py) are correctly configured for your needs.

To start training:

``python src/training/train.py``

Tensorboard
^^^^^^^^^^^
This project uses a tensorboard instance to log and visualize relevant information.
To open a board, make sure to have the `tensorboard` python package installed, type the following command on a terminal:

``tensorboard --logdir [logdir/yourlogs]``

Finally, the command is going to return a localhost address, open it with a web browser.

The location of the logs is controlled the variable `SUMMARIES_DIR`, see the settings module.


StyleGAN
--------
The architecture is described in detail in the paper by Tero Karras et al. (DOI: 10.1109/TPAMI.2020.2970919).
It consists of a generator (the StyleGAN network) and a discriminator/critic.
For more information on the discriminator/critic, refer to the following paper: Tero Karras et al. (DOI: 10.48550/ARXIV.1710.10196).

Loss
^^^^
The loss implemented here is the Wasserstein Loss.
If you want to change the implementation, besides changing the training process you may also need to make changes directly to the models.


Documentation
-------------
To delete the generated files by sphinx, inside the docs/ folder run:
``make clean``
