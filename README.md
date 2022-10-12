# StyleGAN Pytorch
Easy to read implementation of StyleGAN in Pytorch.
### Requirements:
- pytorch
- tensorboard
- sphinx (for documentation)

## Training
#### Run (while on the root folder):
  ``python src/training/train.py``

#### Visualizing with tensorboard:
    tensorboard --logdir=summaries/[name]

#### Based heavily on the works of:
- Tero Karras et al. (DOI: 10.1109/TPAMI.2020.2970919)
- Sharon Zhou et al. (https://www.coursera.org/specializations/generative-adversarial-networks-gans)
- SiskonEmilia (https://github.com/SiskonEmilia/StyleGAN-PyTorch)
