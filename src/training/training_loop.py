import gc
from random import choice
from numpy import linspace
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchsummary import summary
from src.settings import (
    SHOW_NETWORK_SUMMARY, SHOW_EVERY_BATCH, BASE_DIM,
    SAVE_EVERY_BATCHES, RESOLUTIONS, RES_BATCH_SIZE, ALPHA_STEPS
)
from src.training.train_batch import train_batch
from src.device import device


class Resolution:
    def __init__(self, res, dataset, init_training_vars=None):
        if res in RESOLUTIONS:
            self.res = res
            self.count_batch = 0
            # self.dataset_split = dataset_split_large if self.res >= 256 else dataset_split_small
            self.dataset_split = self._split_dataset(dataset, 30)
            self.save_every_batches = SAVE_EVERY_BATCHES[str(self.res)]
            self.batch_size = RES_BATCH_SIZE[str(self.res)]
            self.transform = self._get_transform()
            if init_training_vars:
                self.start_alpha = init_training_vars['start_alpha']
                self.alpha_steps_completed = init_training_vars['alpha_steps_completed']
            else:
                self.start_alpha = 0.001
                self.alpha_steps_completed = 0
        else:
            raise ValueError('Resolution not valid. See settings.')

    def _get_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.res),          # Resize to the same size
            transforms.CenterCrop(self.res),      # Crop to get square area
            transforms.RandomHorizontalFlip(),  # Increase number of samples
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5),)])
        return transform

    def _split_dataset(self, dataset, split_dataset_in):
        """
        Divides 'dataset' into 'split_dataset_in' parts. All of them are of equal
        size except for the last one, which is equal or larger in size than the rest.
        :returns: A random split of the dataset
        """
        dataset_split_lengths = [len(dataset) // split_dataset_in] * (split_dataset_in - 1)
        dataset_split_lengths.append(len(dataset) - sum(dataset_split_lengths))
        dataset_split = random_split(dataset, dataset_split_lengths)
        print('dataset split lens: {}'.format(' '.join(map(lambda x: str(len(x)), dataset_split))))
        return dataset_split


class TrainingLoop:
    def __init__(self, init_training_vars, stgan, disc, g_optim, d_optim, dataset):
        self.stgan = stgan
        self.disc = disc
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.dataset = dataset
        self.stgan.train()
        self.disc.train()
        self.start_res = init_training_vars['start_res']
        if self.start_res not in RESOLUTIONS:
            raise ValueError('Resolution not valid. See settings.')
        self.write_batch_tensorboard = SHOW_EVERY_BATCH * 2
        self.count_batch_global = 0
        self.resolutions_list = self._get_resolutions_list()
        self.res_idx = 0
        self.resolution = Resolution(self.start_res, self.dataset, init_training_vars)
        self.samples_per_res = len(self.resolution.dataset_split[0]) * ALPHA_STEPS

    # @show_network_summary_dec(SHOW_NETWORK_SUMMARY)
    def run(self):
        if SHOW_NETWORK_SUMMARY:
            self.show_network_summary()
        while self.res_idx < len(self.resolutions_list):
            print(f'Beginning of res: {self.resolution.res}')
            self.dataset.transform = self.resolution.transform
            self.disc.start_at_resolution = self.resolution.res
            self.stgan.stop_at_resolution = self.resolution.res
            alpha_steps_left = ALPHA_STEPS - self.resolution.alpha_steps_completed
            for alpha in linspace(self.resolution.start_alpha, 1, num=alpha_steps_left):
                self.stgan.alpha = alpha
                self.disc.alpha = alpha
                dataset_chunk = choice(self.resolution.dataset_split)
                batch_size = min(len(dataset_chunk), self.resolution.batch_size)
                dataloader = DataLoader(
                    dataset_chunk,
                    batch_size=batch_size,
                    shuffle=True
                )
                for images in dataloader:
                    if self.count_batch_global % 50:
                        gc.collect()  # collect garbage
                    images = images[0]  # excluding labels
                    images = images.to(device)
                    train_batch_result = train_batch(
                        self.stgan,
                        self.disc,
                        self.g_optim,
                        self.d_optim,
                        images
                    )
                    fake_images, gen_loss, disc_loss, real_pred, fake_pred = train_batch_result
                    if self.count_batch_global % SHOW_EVERY_BATCH == 0:
                        print(
                            'res: {}. alpha: {}. count batch: {}. gen_loss: {}. disc_loss: {}'.format(
                                self.resolution.res, alpha,
                                self.count_batch_global,
                                gen_loss.mean(),
                                disc_loss.mean()
                            )
                        )

                    self.count_batch_global += 1
                    self.resolution.count_batch += 1
                self.alpha_steps_completed += 1
            self.end_resolution()

    def end_resolution(self):
        self.res_idx += 1
        if self.res_idx < len(self.resolutions_list):
            next_res = self.resolutions_list[self.res_idx]
            self.resolution = Resolution(next_res, self.dataset)

    def show_network_summary_dec(self, func, show=True):
        def inner():
            if show:
                self.disc.start_at_resolution = 8  # needed for the summary
                print('--- Discriminator Summary ---')
                print(summary(self.disc, input_size=(3, 8, 8)))
                print('--- Stylegan Summary ---')
                print(summary(self.stgan, input_size=[(BASE_DIM,), (BASE_DIM,)]))
            func()
        return inner

    def show_network_summary(self):
        self.disc.start_at_resolution = 8  # needed for the summary
        print('--- Discriminator Summary ---')
        print(summary(self.disc, input_size=(3, 8, 8)))
        print('--- Stylegan Summary ---')
        print(summary(self.stgan, input_size=[(BASE_DIM,), (BASE_DIM,)]))

    def _get_resolutions_list(self):
        """
        :return: a list of valid resolutions that are greater or equal
        to 'start_res'
        """
        new_resolutions = list()
        for res in RESOLUTIONS:
            if res >= self.start_res:
                new_resolutions.append(res)
        return new_resolutions
