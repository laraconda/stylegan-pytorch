# StyleGAN Pytorch
Easy to read implementation of StyleGAN in Pytorch.
### Requirements:
- pytorch
- tensorboard

## Training
#### Run (while on the root folder):
    python -m src.training.train

#### Visualizing with tensorboard:
    tensorboard --logdir=summaries/[name]

### Important
This project does not yet support single channel image datasets. It should not be difficult to change it for your specific needs though.

#### Based heavily on the works of:
- Tero Karras et al. (DOI: 10.1109/TPAMI.2020.2970919)
- Sharon Zhou et al. (https://www.coursera.org/specializations/generative-adversarial-networks-gans)
- SiskonEmilia (https://github.com/SiskonEmilia/StyleGAN-PyTorch)
