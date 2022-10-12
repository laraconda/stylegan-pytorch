"""
Definition of the main component of the training of the networks, the `TrainingLoop` class.
"""

import gc
from random import choice
from numpy import linspace
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchsummary import summary
from settings import (
    SHOW_NETWORK_SUMMARY, PRINT_BATCH_EVERY_X_ITERATIONS, BASE_DIM,
    SAVE_EVERY_X_ITERATIONS, RESOLUTIONS, RES_BATCH_SIZE, ALPHA_STEPS,
    N_SAMPLES_RES, COLLECT_GARBAGE_EVERY_X_ITERATIONS,
    WRITE_BATCH_EVERY_X_ITERATIONS, SUMMARIES_DIR, DATASET_CHANNELS
)
from training.train_batch import train_batch
from training.save import save_checkpoint
from device import device
from mywriter import MyWriter, TrainingBatchInfo


class Resolution:
    """
    Contains information about a training resolution. Used by `TrainingLoop`.

    Attributes
    ----------
    res: int
        Numerical resolution
    resolution_step: int
        Number of batches that have trained the networks during this resolution
    save_every_x_iterations: int
        Frequency of saving a checkpoint. It varies depending on the numerical resolution.
    batch_size: int
        Size of the batches used to train the networks. It varies depending on
        the numerical resolution.
    start_alpha: float
        Starting point for the alpha steps on this resolution. It's zero if the
        networks have not been trained on this resolution.
    alpha_steps_completed: int
        Number of discrete alpha steps completed on the training of this resolution
        so far.

    Methods
    -------
    get_dataset_transform()
        Returns a dataset transform that among other things, resizes images to `res`.
    """
    def __init__(self, res, init_training_vars=None):
        """
        Parameters
        ----------
        res: int
            Numerical resolution
        init_training_vars: dict, optional
            Dict containing variables used to initialize attributes. If its None
            then certain class attributes will be initialized to a default value.

        Raises
        ------
        ValueError
            When the starting resolution specified in init_training_vars is not in
            the `RESOLUTIONS` list.
        """
        if res in RESOLUTIONS:
            self.res = res
            self.resolution_step = 0
            self.save_every_x_iterations = SAVE_EVERY_X_ITERATIONS[str(self.res)]
            self.batch_size = RES_BATCH_SIZE[str(self.res)]
            if init_training_vars:
                self.start_alpha = init_training_vars['start_alpha']
                self.alpha_steps_completed = init_training_vars['alpha_steps_completed']
            else:
                self.start_alpha = 0.001
                self.alpha_steps_completed = 0
        else:
            raise ValueError('Resolution not valid. See settings.')

    def get_dataset_transform(self):
        """
        Returns a dataset transform that among other things, resizes tensor images to `res`.

        Returns
        -------
        transform
            A dataset transform to resize tensor images to `res`.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.res),  # Resize to the same size
            transforms.CenterCrop(self.res),  # Crop to get square area
            transforms.RandomHorizontalFlip(),  # Increase number of samples
            transforms.Normalize([0.5] * DATASET_CHANNELS,
                                 [0.5] * DATASET_CHANNELS,)])
        return transform


class TrainingLoop:
    """
    Starts or continues a training session.

    The training scheme used here has three main loops. First is the progressive
    growing of the training images, second is the alpha used for the smooth introduction
    of blocks that goes from 0 to 1 and third is the dataloader that iterates through
    batches.

    If settings.py specify that a checkpoint is to be used as a starting point
    then the training will continue from that point.

    Many settings that control the behavior of the training and thus of this class
    are defined in settings.py. Some of behaviors are described further down this section.

    This training uses a progressive growing approach, meaning that after training the
    networks on a small initial resolution, the resolution of the training images will
    increase after certain number of batches. The resolutions the training is going to
    train on is defined on `RESOLUTIONS`.

    A new training id is assigned to the training session if the session is not
    initialized from a checkpoint, the training id retreived from the checkpoint is used
    if otherwise.

    Attributes
    ----------
    stgan: StyleGAN
        The generator network.
    disc: Discriminator
        The discriminator network.
    g_optim: Optimizer
        The optimizer of the generator.
    d_optim: Optimizer
        The optimizer of the discriminator.
    dataset: Dataset
        The training dataset.
    dataset_split:
        A split of the dataset into various subsets.
    run_id: string
        The id of the training.
    start_res: int
        The resolution where the training will begin.
    summary_dir: string
        A directory were tensorboard summaries will be written.
    global_step: int
        Number of the current training batch.
    log_filename_suffix: string
        A suffix for the summary name.
    writer: MyWriter
        A helper class to easily write information to the board.
    write_batch_every_x_iterations: int
        Variable to control how often information is written to the board.
    resolutions_list: list
        The resolutions (in order) the networks will be trained on. The first
        resolution not always is the first resolution in `RESOLUTIONS`.
    res_idx: int
        The id (based on `resolutions_list`) of the current resolution.
    resolution: Resolution
        A helper class that contains information about a training resolution.

    Methods
    -------
    start()
        Starts the training process.
    finish()
        Finalizes the training process.
    """
    def __init__(self, init_training_vars, stgan, disc, g_optim, d_optim, dataset):
        """
        Parameters
        ----------
        init_training_vars: dict
            Dict containing necessary variables to start or restart the training.
        stgan: StyleGAN
            Instance of the generator.
        disc: Discriminator
            Discriminator network.
        g_optim: Optimizer
            Optimizer for the generator.
        d_optim: Optimizer
            Optimizer for the discriminator.
        dataset: Dataset
            Training dataset.

        Raises
        ------
        ValueError
            When the starting resolution specified in init_training_vars is not in
            the `RESOLUTIONS` list.
        """
        self.stgan = stgan
        self.disc = disc
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.dataset = dataset
        self.dataset_split = self._split_dataset()
        self.stgan.train()
        self.disc.train()
        self.run_id = init_training_vars['run_id']
        self.start_res = init_training_vars['start_res']
        self.summary_dir = '{}/{}'.format(SUMMARIES_DIR, self.run_id)
        self.global_step = init_training_vars['global_step']
        self.log_filename_suffix = init_training_vars['log_filename_suffix']
        self.writer = MyWriter(
            self.summary_dir, self.log_filename_suffix, self.disc, self.stgan
        )
        if self.start_res not in RESOLUTIONS:
            raise ValueError('Resolution not valid. See settings.')
        self.write_batch_every_x_iterations = PRINT_BATCH_EVERY_X_ITERATIONS * 2
        self.resolutions_list = self._get_resolutions_list()
        self.res_idx = 0
        self.resolution = Resolution(self.start_res, init_training_vars)

    def start(self):
        """
        Starts the training process.

        During the execution of this function, information about the batches is printed
        to the standard output as well as written to a tensorboard instance; to access said
        isntance is necessary to open a web browser and go to the specified location and port.

        The constant `COLLECT_GARBAGE_EVERY_X_ITERATIONS` controls the frequency of garbage
        collection.

        Checkpoints are saved every certain number of batches and it varies depending on the
        current resolution of the training, the constant that controls it is the dict
        `SAVE_EVERY_X_ITERATIONS`.

        """
        if SHOW_NETWORK_SUMMARY:
            self._show_network_summary()
        while self.res_idx < len(self.resolutions_list):
            print(f'Beginning of res: {self.resolution.res}')
            self.dataset.transform = self.resolution.get_dataset_transform()
            self.disc.start_at_resolution = self.resolution.res
            self.stgan.stop_at_resolution = self.resolution.res
            alpha_steps_left = ALPHA_STEPS - self.resolution.alpha_steps_completed
            for alpha in linspace(self.resolution.start_alpha, 1, num=alpha_steps_left):
                self.stgan.alpha = alpha
                self.disc.alpha = alpha
                dataset_chunk = choice(self.dataset_split)
                batch_size = min(len(dataset_chunk), self.resolution.batch_size)
                dataloader = DataLoader(
                    dataset_chunk,
                    batch_size=batch_size,
                    shuffle=True
                )
                for real_images in dataloader:
                    if self.global_step % COLLECT_GARBAGE_EVERY_X_ITERATIONS:
                        gc.collect()  # collect garbage
                    real_images = real_images[0]  # excluding labels
                    real_images = real_images.to(device)
                    train_batch_result = train_batch(
                        self.stgan,
                        self.disc,
                        self.g_optim,
                        self.d_optim,
                        real_images
                    )
                    fake_images, gen_loss, disc_loss, real_pred, fake_pred = train_batch_result
                    training_batch_info = TrainingBatchInfo(
                        fake_images,
                        real_images,
                        gen_loss,
                        disc_loss,
                        real_pred,
                        fake_pred,
                        self.global_step
                    )
                    if (
                        self.global_step %
                        self.resolution.save_every_x_iterations == 0
                    ) and self.resolution.resolution_step > 0:
                        save_checkpoint(
                            self.stgan.state_dict(),
                            self.disc.state_dict(),
                            self.g_optim.state_dict(),
                            self.d_optim.state_dict(),
                            self.resolution.res,
                            self.run_id,
                            self.global_step,
                            self.resolution.alpha_steps_completed
                        )
                    if self.global_step % PRINT_BATCH_EVERY_X_ITERATIONS == 0:
                        self._log_batch(alpha, training_batch_info)
                    if (
                        self.global_step % WRITE_BATCH_EVERY_X_ITERATIONS == 0 or
                        (
                            self.global_step %
                            self.resolution.save_every_x_iterations == 0 and
                            self.global_step > 0
                        )  # writing on save too
                    ):
                        self.writer.write_batch(training_batch_info)

                    self.global_step += 1
                    self.resolution.resolution_step += 1
                self.resolution.alpha_steps_completed += 1
            self._end_resolution()
        self.stop()

    def finish(self):
        self.writer.close()

    def _log_batch(self, alpha, training_batch_info):
        """
        Prints to the stdout information about the current batch defined in `training_batch_info`.

        Parameters
        ----------
        alpha: float
            The alpha step of the training.
        training_batch_info: TrainingBatchInfo
            An object containing information about the current batch.
        """
        print(
            'res: {}. alpha: {}. global batch: {}. gen loss: {}. disc loss: {}'
            .format(
                self.resolution.res,
                alpha,
                self.global_step,
                training_batch_info.gen_loss.mean(),
                training_batch_info.disc_loss.mean()
            )
        )

    def _end_resolution(self):
        """
        Ends the training process for the current resolution.

        If a larger resolution is defined in `RESOLUTIONS`, then it prepares to start training
        the networks with image of that resolution.
        """
        print(f'End of res {self.resolution.res}')
        self.res_idx += 1
        if self.res_idx < len(self.resolutions_list):
            next_res = self.resolutions_list[self.res_idx]
            self.resolution = Resolution(next_res)

    def _show_network_summary(self):
        """
        Shows a summary of the networks.

        Prints a dissection of the two main networks, showing its layers, activations,
        number of trainable parameters, etc.
        """
        self.disc.start_at_resolution = 8  # needed for the summary
        print('--- Discriminator Summary ---')
        print(summary(self.disc, input_size=(3, 8, 8)))
        print('--- Stylegan Summary ---')
        print(summary(self.stgan, input_size=[(BASE_DIM,), (BASE_DIM,)]))

    def _split_dataset(self):
        """
        Divides `dataset` into `split_dataset_in` parts. All of them are of equal
        size except for the last one, which is equal or larger in size than the rest.

        Returns
        -------
        list-like
            A random split of the dataset.
        """
        lendataset = len(self.dataset)
        split_dataset_in = ALPHA_STEPS * lendataset // N_SAMPLES_RES
        dataset_split_lengths = [lendataset // split_dataset_in] * (split_dataset_in - 1)
        dataset_split_lengths.append(lendataset - sum(dataset_split_lengths))
        dataset_split = random_split(self.dataset, dataset_split_lengths)
        print(
            '''Dataset splat in chunks of len: {}.
                This split is controled by the constants:
                ALPHA_STEPS and N_SAMPLES_RES
            '''.format(' '.join(map(lambda x: str(len(x)), dataset_split)))
        )
        return dataset_split

    def _get_resolutions_list(self):
        """
        Returns the ordered resolutions the training session is going to go through.

        If a training session is continued from a checkpoint then the starting resolution
        could be any of the valid resolutions defined in `RESOLUTIONS` (see settings.py).

        Returns
        -------
        list
            A list of valid resolutions that are greater or equal to `start_res`.
        """
        new_resolutions = list()
        for res in RESOLUTIONS:
            if res >= self.start_res:
                new_resolutions.append(res)
        return new_resolutions
