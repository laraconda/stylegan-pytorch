"""
Definition of functions used to train both networks.
"""

import torch
from settings import BASE_DIM, GRADIENT_PENALTY_LAMBDA
from loss import gradient_penalty
from device import device


def train_disc_batch(disc, real_images, d_optim, stgan, batch_size):
    """
    Trains the discriminator network with a batch of images.

    This function changes the trainable parameters of `disc`.

    Parameters
    ----------
    disc: Discriminator
        The instance of Discriminator to train.
    real_images: tensor
        A batch of real images from the training dataset.
    d_optim: Optimizer
        The optimizer for the discriminator.
    stgan: StyleGAN
        The generator network used to generate fake images.
    batch_size: int
        The number of images in the batch.

    Returns
    -------
    tuple
        This tuple contains the loss of the discriminator during this batch, the predictions
        of the discriminator for the real images and the prediction of the discriminator
        for a set of fake images generated inside the function.
    """
    disc.zero_grad()
    real_pred = disc(real_images)
    with torch.no_grad():
        z1 = torch.randn((batch_size, BASE_DIM), device=device)
        z2 = torch.randn((batch_size, BASE_DIM), device=device)
        fake_images = stgan(z1, z2)

    fake_pred = disc(fake_images)
    gp = gradient_penalty(disc, real_images, fake_images)
    loss = torch.mean(fake_pred) - torch.mean(real_pred) + GRADIENT_PENALTY_LAMBDA * gp
    loss.backward()
    d_optim.step()

    del fake_images, z1, z2
    return loss, real_pred, fake_pred


def train_gen_batch(stgan, g_optim, disc, batch_size):
    """
    Trains the generator with a batch of images generated by the network.

    This function changes the trainable parameters of the generator network.

    Parameters
    ----------
    stgan: StyleGAN
        The generator network.
    g_optim: Optimizer
        The optimizer for `stgan`.
    disc: Discriminator
        The discriminator network.
    batch_size: int
        The number of images to be produced by the generator.

    Returns
    -------
    tuple
        Contains the tensor containing the fake images generated by the generator
        and the loss of that network during this batch.
    """
    stgan.zero_grad()
    z1 = torch.randn((batch_size, BASE_DIM), device=device)
    z2 = torch.randn((batch_size, BASE_DIM), device=device)
    fake_images = stgan(z1, z2)
    fake_pred = disc(fake_images)
    loss = -torch.mean(fake_pred)
    loss.backward()
    g_optim.step()

    del z1, z2, fake_pred
    return fake_images, loss


def train_batch(stgan, disc, g_optim, d_optim, real_images):
    """
    Trains both the discriminator and the generator networks.

    Parameters
    ----------
    stgan: StyleGAN
        The generator network.
    disc: Discriminator
        The discriminator netowrk.
    g_optim: Optimizer
        The optimizer for the generator.
    d_optim: Optimizer
        The optimizer for the discirminator.
    real_images: tensor
        A batch of real images from the training dataset.

    Returns
    -------
    tuple
        Contains a batch of fake images generated by the generator, the loss of the generator,
        the loss of the discriminator, the predictions for the batch of real images by
        the discriminator and the prediction for the fake images.
    """
    batch_size = len(real_images)
    disc_loss, real_pred, fake_pred = train_disc_batch(
        disc,
        real_images,
        d_optim,
        stgan,
        batch_size
    )
    fake_images, gen_loss = train_gen_batch(stgan, g_optim, disc, batch_size)
    return fake_images, gen_loss, disc_loss, real_pred, fake_pred
