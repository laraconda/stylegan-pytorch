# StyleGAN Pytorch
Easy to read and documented implementation of StyleGAN in Pytorch.

### Requirements:
- torch
- torchvision
- tensorboard
- torchsummary

For documentation:
- sphinx
- numpydoc
- sphinx-rtd-theme

## Training
#### Run:
Before starting the training process, make sure the settings (found in src/settings.py) are correctly configured for your needs.

    python src/training/train.py

#### Visualizing with tensorboard:
On a new terminal:

    tensorboard --logdir=summaries/[summary-name]

## Generate documentation
Inside the /doc folder:

    make html

To delete the html files:

    make clean

## Important
This project only works with square images.

This is an implementation of StyleGAN 1.

## Based on the works of:
- Tero Karras et al. (DOI: 10.1109/TPAMI.2020.2970919)
- Sharon Zhou et al. (https://www.coursera.org/specializations/generative-adversarial-networks-gans)
- SiskonEmilia (https://github.com/SiskonEmilia/StyleGAN-PyTorch)
