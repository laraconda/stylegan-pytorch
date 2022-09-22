import torch
from src.settings import BASE_DIM, GRADIENT_PENALTY_LAMBDA
from src.loss import gradient_penalty
from src.device import device


def train_disc_batch(disc, real_images, d_optim, stgan, batch_size):
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
