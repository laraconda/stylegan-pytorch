import torch
from src.device import device


def get_gradient(disc, real_images, fake_images):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    :crit: the critic model
    :real_images: a batch of real images
    :fake_images: a batch of fake images
    :epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    :returns: gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    epsilon = torch.rand((real_images.size(0), 1, 1, 1), device=device, requires_grad=True)
    mixed_images = real_images * epsilon + fake_images * (1 - epsilon)

    # Calculate the disc's scores on the mixed images
    mixed_scores = disc(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(disc, real_images, fake_images):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    :crit: the critic model
    :real_images: a batch of real images
    :fake_images: a batch of fake images
    :returns: penalty: the gradient penalty
    '''
    gradient = get_gradient(disc, real_images, fake_images)
    # Flatten the gradients so that each row captures one image
    grad_size = gradient.size()
    gradient = gradient.reshape((grad_size[0], grad_size[1] * grad_size[2] * grad_size[3]))
    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean(torch.pow(gradient_norm - torch.ones_like(gradient_norm), 2))
    return penalty
