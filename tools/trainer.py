import torch
import pytorch_lightning as pl
from tools import dataset
from utils.conf_utils import instantiate_from_config, instantiate_from_config_with_params
from utils.img_utils import shave, rgb_to_ycbcr
from utils.ssim import SSIM
from torchmetrics import Metric
from tools.loggers import ImageLogger
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
import cv2
import numpy as np

class MeanSSIM(Metric):
    def __init__(self, channel_num, data_range=1.0):
        super(MeanSSIM, self).__init__()
        self.add_state("ssims", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.ssim_calc = SSIM(data_range=data_range,
                              channel=channel_num)

    def update(self, hr_image, sr_image):
        self.ssims += self.ssim_calc(hr_image, sr_image)
        self.total += 1

    def compute(self):
        val = self.ssims.float() / self.total
        return val


class MeanPSNR(Metric):
    def __init__(self):
        super(MeanPSNR, self).__init__()
        self.add_state("psnrs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, hr_image, sr_image):
        self.psnrs += torch.mean(10 * torch.log10(1 / (torch.mean(torch.square(hr_image - sr_image), (1, 2, 3)))))
        self.total += 1

    def compute(self):
        val = self.psnrs.float() / self.total
        return val

class SRTrainer(pl.LightningModule):
    def __init__(self, conf):
        super(SRTrainer, self).__init__()
        self.conf = conf
        self.scale = self.conf['data']['train']['scale']
        self.lr = self.conf['trainer']['base_lr_rate']
        self.num_epochs = self.conf['trainer']['num_epochs']
        self.use_Y = self.conf['trainer']['use_Y_channel_in_val']

        self.log_images = self.conf['loggers']['log_images']
        if self.log_images:
            self.image_logger = ImageLogger()

        self.psnr = MeanPSNR()
        self.best_psnr = 0.0
        self.ssim = MeanSSIM(channel_num=1 if self.conf['trainer']['use_Y_channel_in_val'] else 3,
                             data_range=1.0)
        self.best_ssim = 0.0
        self.network = instantiate_from_config(self.conf['network'])

        self.save_hyperparameters()  ## Save HyperParameters to self.hparrams

        self.validation_outputs = []

    def train_dataloader(self):
        ds = dataset.SRDataset(conf_data=self.conf['data']['train'])
        loader = torch.utils.data.DataLoader(dataset=ds, **self.conf['loader']['train'])
        # loader = torch.utils.od_images.DataLoader(dataset=ds, collate_fn=self.custom_collate_fn, **self.conf['loader']['train'])
        return loader

    def val_dataloader(self):
        ds = dataset.SRDataset(conf_data=self.conf['data']['val'])
        loader = torch.utils.data.DataLoader(dataset=ds, **self.conf['loader']['val'])
        # loader = torch.utils.od_images.DataLoader(dataset=ds, collate_fn=self.custom_collate_fn, **self.conf['loader']['val'])
        return loader


    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.99, 0.999))
        scheduler = instantiate_from_config_with_params(config=self.conf['trainer']['lr_scheduler'],
                                                        additional_params={'optimizer': optimizer})
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1}

        return [optimizer], [lr_scheduler]

    def training_step(self, data):
        lr_images = data['img_lr']
        hr_images = data['img_hr']

        lr_images = lr_images.type(torch.float32) / 255.
        hr_images = hr_images.type(torch.float32) / 255.

        lr_images = (lr_images - 0.5) / 0.5
        hr_images = (hr_images - 0.5) / 0.5

        sr_images = self.forward(lr_images)
        # rec_loss = self.mse_loss(sr_images, hr_images)
        # rec_loss = self.ssim_loss(sr_images, hr_images)
        rec_loss = self.psnr_loss(sr_images, hr_images)
        self.log("loss", rec_loss, on_step=True, sync_dist=True, prog_bar=True)

        return rec_loss

    def validation_step(self, data, i):
        lr_images = data['img_lr']
        hr_images = data['img_hr']

        lr_images = lr_images.type(torch.float32) / 255.
        hr_images = hr_images.type(torch.float32) / 255.

        lr_images = (lr_images - 0.5) / 0.5
        sr_images = self.forward(lr_images)
        sr_images = sr_images / 2 + 0.5

        self.update_val_metrics(sr_images, hr_images)

        ########for showing SR, debugging ##########是伐
        # while True:
        #     sr_img = tensor2img(sr_images)
        #     cv2.imshow("Super-Resolved Image", sr_img)
        #     key = cv2.waitKey(0) & 0xFF
        #     # print(key)  # Debug: 打印按键的值
        #     if key == ord('q'):
        #         break
        # # cv2.destroyAllWindows()
        ###################

        # Store outputs for potential image logging
        self.validation_outputs.append(sr_images)
        return sr_images

    def on_validation_epoch_end(self):
        val_psnr = self.psnr.compute().detach().cpu().item()
        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            if self.log_images:
                pass
                self.image_logger(self.validation_outputs)
        self.log("best_val_psnr", self.best_psnr, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log("val_psnr", val_psnr, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.psnr.reset()

        val_ssim = self.ssim.compute()
        if val_ssim > self.best_ssim:
            self.best_ssim = val_ssim
        self.log("best_val_ssim", self.best_ssim, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log("val_ssim", val_ssim, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.ssim.reset()

        # Clear the validation outputs after logging
        self.validation_outputs.clear()

    def update_val_metrics(self, sr_images, hr_images):
        shaved_sr = shave(sr_images, self.scale)
        shaved_hr = shave(hr_images, self.scale)
        if self.use_Y:
            shaved_sr = rgb_to_ycbcr(shaved_sr)
            shaved_hr = rgb_to_ycbcr(shaved_hr)
        self.psnr.update(torch.clamp(shaved_sr, 0, 1),
                         torch.clamp(shaved_hr, 0, 1))
        self.ssim.update(torch.clamp(shaved_sr, 0, 1),
                         torch.clamp(shaved_hr, 0, 1))

    @staticmethod
    def mse_loss(I1, I2):
        return torch.mean(torch.square(I1 - I2))

    def ssim_loss(self, sr_img, hr_img):
        ssim_index = ssim(sr_img, hr_img, data_range=1.0)
        loss = 1 - ssim_index  # SSIM loss is 1 - SSIM value
        return loss

    def psnr_loss(self, output: torch.Tensor, target: torch.Tensor, max_val= 1.0) -> torch.Tensor:
        mse = F.mse_loss(output, target, reduction='mean')
        if mse == 0:
            # When MSE is zero, the PSNR is infinite.
            return torch.tensor(0.0, device=output.device)

        psnr = 10 * torch.log10(max_val ** 2 / mse)
        # Add a small constant to avoid division by zero
        epsilon = 1e-8
        # Convert PSNR to a loss value where smaller is better
        # We can take the inverse plus a small constant to ensure it's always positive
        loss = 1 / (psnr + epsilon)
        return loss

    def forward(self, inputs):
        return self.network(inputs)

def tensor2img(x):
    # Convert the tensor to a NumPy array for displaying
    img = x.squeeze().cpu().numpy().transpose(1, 2, 0)  # Assuming the tensor is (C, H, W)
    img = np.clip(img, 0, 1)  # Ensure the values are in the [0, 1] range
    img = (img * 255).astype('uint8')  # Scale to 0-255 for display
    # Convert from RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img