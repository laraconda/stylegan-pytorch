"""
Implementation of the MyWriter class which defines what is written to the active
tensorboard board. The training_batch_info class is used as a bundle of information related
to a certain batch.
"""

from torch import tensor, stack
from device import device
from torch.utils.tensorboard import SummaryWriter
from models.generator.networks import StyleGAN
from models.discriminator import Discriminator
from settings import (
    MAPPN_NETWORK_LR, EPS, GRADIENT_PENALTY_LAMBDA, STD, STGAN_LRATE,
    DISC_LRATE, WEIGHT_DECAY, N_SAMPLES_RES, DATASET_CHANNELS
)


class MyWriter:
    """
    Implements methods to write training information to a tensorboard instance.

    Attributes
    ----------
    writer: SummaryWriter
        Tensorboard writer.
    disc: Discriminator
        The discriminator network. Used to plot its parameters.
    stgan: StyleGAN
        The generator network. Used to plot its parameters.

    Methods
    -------
    close()
        Closes the `writer` instance. It flushes it before.
    write_batch(training_batch_info)
        Writes the information of a training batch on a tensorboard instance (`writer`).
    """
    def __init__(self, summary_dir, log_filename_suffix, disc, stgan):
        """
        Parameters
        ----------
        summary_dir: string
        log_filename_suffix: string
        disc: Discriminator
        stgan: StyleGAN
        """
        self.writer = SummaryWriter(summary_dir, log_filename_suffix)
        self.disc = disc
        self.stgan = stgan

    def close(self):
        """
        Closes the `writer` instance. It flushes it before.
        """
        self.writer.flush()
        self.writer.close()

    def _unnormalize(self, image):
        """
        Unnormalizes a tensor image.

        Parameters
        ----------
        image: tensor
            The image to be unnormalized.

        Returns
        -------
        tensor
            The unnormalized image.
        """
        MEAN = tensor([0.5] * DATASET_CHANNELS).to(device)
        STD = tensor([0.5] * DATASET_CHANNELS).to(device)
        return image * STD[:, None, None] + MEAN[:, None, None]

    def write_batch(self, training_batch_info):
        """
        Writes the information of a training batch on a tensorboard instance (`writer`).

        Parameters
        ----------
        training_batch_info: TrainingBatchInfo
            Batch information.
        """
        self._add_param_histograms(self.disc, training_batch_info.global_step)
        self._add_param_histograms(self.stgan, training_batch_info.global_step)
        fake_images_un = stack([self._unnormalize(x) for x in training_batch_info.fake_images])
        images_un = stack([self._unnormalize(x) for x in training_batch_info.real_images])
        self.writer.add_images(
            'fake images', fake_images_un, global_step=training_batch_info.global_step
        )
        self.writer.add_images(
            'real images', images_un, global_step=training_batch_info.global_step
        )
        self.writer.add_text(
            'fake tensor',
            str(training_batch_info.fake_images[0]),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text(
            'real tensor',
            str(training_batch_info.real_images[0]),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text(
            'real pred',
            str(training_batch_info.real_pred),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text(
            'fake pred',
            str(training_batch_info.fake_pred),
            global_step=training_batch_info.global_step
        )
        self.writer.add_scalar(
            'real pred avg (abs)',
            training_batch_info.real_pred.abs().mean(),
            global_step=training_batch_info.global_step
        )
        self.writer.add_scalar(
            'fake pred avg (abs)',
            training_batch_info.fake_pred.abs().mean(),
            global_step=training_batch_info.global_step
        )
        self.writer.add_graph(self.disc, training_batch_info.real_images)
        if training_batch_info.global_step % 1000:
            self._write_hyperparams(training_batch_info)
        self.writer.flush()

    def _write_hyperparams(self, training_batch_info):
        """
        Writes on a tensorboard instance (`writer`) the hyperparameters used to train a batch.

        Parameters
        ----------
        training_batch_info: TrainingBatchInfo
            Batch information.
        """
        self.writer.add_text(
            'lr stgan',
            str(STGAN_LRATE),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text(
            'lr disc',
            str(DISC_LRATE),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text(
            'weight_decay',
            str(WEIGHT_DECAY),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text(
            'mappn network lr',
            str(MAPPN_NETWORK_LR),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text(
            'samples per res',
            str(N_SAMPLES_RES),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text('STD', str(STD), global_step=training_batch_info.global_step)
        self.writer.add_text(
            'gradient penalty lambda',
            str(GRADIENT_PENALTY_LAMBDA),
            global_step=training_batch_info.global_step
        )
        self.writer.add_text('eps', str(EPS), global_step=training_batch_info.global_step)

    def _add_param_histograms(self, module, global_step):
        """
        Adds the weights of every parameter in `module` to a tensorboard instance as histograms.

        If module is not an instance of StyleGAN or Discriminator then the function
        does nothing.

        Parameters
        ----------
        module: {StyleGAN, Discriminator}
            Module whose parameters are going to be written.
        global_step: int
            Number of the current training batch.
        """

        def get_block_id(name):
            """
            Out of the name of a block, this function retrieves the id of the block.

            The 'id of the block' is in relation to it's place on a ModuleList. See the models
            package for more information.

            Parameters
            ----------
            name: string
                The name of a block of an instance of StyleGAN or Discriminator.

            Returns
            -------
            int
                The id of the block.
            """
            for i in name:
                if i.isdigit():
                    return int(i)

        for name, weight in module.named_parameters():
            if weight.grad is not None:
                if isinstance(module, StyleGAN) or isinstance(module, Discriminator):
                    block_id = get_block_id(name)
                    if isinstance(module, StyleGAN):
                        res = StyleGAN.blocktores(block_id)
                    if isinstance(module, Discriminator):
                        res = Discriminator.blocktores(block_id)
                    self.writer.add_histogram(
                        f'{module.__class__.__name__}-{name} ({res}px)',
                        weight,
                        global_step
                    )
                    self.writer.add_histogram(
                        f'{module.__class__.__name__}-{name}.grad ({res}px)',
                        weight.grad,
                        global_step
                    )


class TrainingBatchInfo:
    """
    Object storing information about a training bach.

    Attributes
    ----------
    fake_images: tensor
        Batch of images generated by the generator network.
    real_images: tensor
        Batch of images from the training dataset.
    gen_loss: number
        Loss of the generator.
    disc_loss: number
        Loss of the discriminator.
    real_pred: tensor
        Predicitons by the discriminator on the batch of real images.
    fake_pred: tensor
        Predicitons by the generator on the batch of real images.
    global_step: int
        Number of the current training batch.
    """
    def __init__(
        self,
        fake_images,
        real_images,
        gen_loss,
        disc_loss,
        real_pred,
        fake_pred,
        global_step
    ):
        """
        Parameters
        ----------
        fake_images: tensor
            Batch of images generated by the generator network.
        real_images: tensor
            Batch of images from the training dataset.
        gen_loss: number
            Loss of the generator.
        disc_loss: number
            Loss of the discriminator.
        real_pred: tensor
            Predicitons by the discriminator on the batch of real images.
        fake_pred: tensor
            Predicitons by the generator on the batch of real images.
        global_step: int
            Number of the current training batch.
        """
        self.fake_images = fake_images
        self.real_images = real_images
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.real_pred = real_pred
        self.fake_pred = fake_pred
        self.global_step = global_step
