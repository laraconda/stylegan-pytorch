#  %load_ext tensorboard
from torch import tensor, stack
from src.device import device
from torch.utils.tensorboard import SummaryWriter
from src.models.generator.networks import StyleGAN
from src.models.discriminator import Discriminator
from src.settings import (
    MAPPN_NETWORK_LR, EPS, GRADIENT_PENALTY_LAMBDA, STD, STGAN_LRATE,
    DISC_LRATE, WEIGHT_DECAY, N_SAMPLES_RES
)


class MyWriter:
    def __init__(self, summary_dir, log_filename_suffix, disc, stgan):
        self.writer = SummaryWriter(summary_dir, log_filename_suffix)
        self.disc = disc
        self.stgan = stgan

    def _unnormalize(self, image):
        MEAN = tensor([0.5, 0.5, 0.5]).to(device)
        STD = tensor([0.5, 0.5, 0.5]).to(device)
        return image * STD[:, None, None] + MEAN[:, None, None]

    def write_batch(self, batchinfo):
        self._add_param_histograms(self.disc, batchinfo.global_step)
        self._add_param_histograms(self.stgan, batchinfo.global_step)
        fake_images_un = stack([self._unnormalize(x) for x in batchinfo.fake_images])
        images_un = stack([self._unnormalize(x) for x in batchinfo.images])
        self.writer.add_images(
            'fake images', fake_images_un, global_step=batchinfo.global_step
        )
        self.writer.add_images('real images', images_un, global_step=batchinfo.global_step)
        self.writer.add_text(
            'fake tensor',
            str(batchinfo.fake_images[0]),
            global_step=batchinfo.global_step
        )
        self.writer.add_text(
            'real tensor', str(batchinfo.images[0]), global_step=batchinfo.global_step
        )
        self.writer.add_text(
            'real pred', str(batchinfo.real_pred), global_step=batchinfo.global_step
        )
        self.writer.add_text(
            'fake pred', str(batchinfo.fake_pred), global_step=batchinfo.global_step
        )
        self.writer.add_scalar(
            'real pred avg (abs)',
            batchinfo.real_pred.abs().mean(),
            global_step=batchinfo.global_step
        )
        self.writer.add_scalar(
            'fake pred avg (abs)',
            batchinfo.fake_pred.abs().mean(),
            global_step=batchinfo.global_step
        )
        self.writer.add_graph(self.disc, batchinfo.images)
        if batchinfo.global_step % 1000:
            self._write_hyperparams(batchinfo)
        self.writer.flush()

    def _write_hyperparams(self, batchinfo):
        self.writer.add_text('lr stgan', str(STGAN_LRATE), global_step=batchinfo.global_step)
        self.writer.add_text('lr disc', str(DISC_LRATE), global_step=batchinfo.global_step)
        self.writer.add_text(
            'weight_decay',
            str(WEIGHT_DECAY),
            global_step=batchinfo.global_step
        )
        self.writer.add_text(
            'mappn network lr',
            str(MAPPN_NETWORK_LR),
            global_step=batchinfo.global_step
        )
        self.writer.add_text(
            'samples per res',
            str(N_SAMPLES_RES),
            global_step=batchinfo.global_step
        )
        self.writer.add_text('STD', str(STD), global_step=batchinfo.global_step)
        self.writer.add_text(
            'gradient penalty lambda',
            str(GRADIENT_PENALTY_LAMBDA),
            global_step=batchinfo.global_step
        )
        self.writer.add_text('eps', str(EPS), global_step=batchinfo.global_step)

    def _add_param_histograms(self, module, global_step):
        """
        Adds the weights of every parameter in 'module' to
        a tensorboard instance as histograms
        """

        def get_block_id(name):
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


class BatchInfo:
    def __init__(
        self,
        fake_images,
        images,
        gen_loss,
        disc_loss,
        real_pred,
        fake_pred,
        global_step
    ):
        self.fake_images = fake_images
        self.images = images
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.real_pred = real_pred
        self.fake_pred = fake_pred
        self.global_step = global_step
