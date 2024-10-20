'''
Evaluate the psnr and ssim

'''
import numpy as np
import torch
from utils.ssim import SSIM
from utils import conf_utils
from tools import dataset
from tools import trainer
from utils.img_utils import shave, rgb_to_ycbcr
from torchmetrics import Metric
from tqdm import tqdm
import cv2
from models.arch import RTSR as standard_arch
from models.mini_arch import RTSR as  mini_arch

class MeanPSNR(Metric):
    def __init__(self):
        super(MeanPSNR, self).__init__()
        self.add_state("psnrs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, sr_image, hr_image):
        mse = torch.mean(torch.square(sr_image - hr_image), dim=(1, 2, 3))
        psnr = 10 * torch.log10(1.0 / mse)
        self.psnrs += torch.mean(psnr)
        self.total += 1

    def compute(self):
        return self.psnrs.float() / self.total

class MeanSSIM(Metric):
    def __init__(self, channel_num, data_range=1.0):
        super(MeanSSIM, self).__init__()
        self.add_state("ssims", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.ssim_calc = SSIM(data_range=data_range, channel=channel_num)

    def update(self, sr_image, hr_image):
        self.ssims += self.ssim_calc(sr_image, hr_image)
        self.total += 1

    def compute(self):
        return self.ssims.float() / self.total

class MyPSNR:
    def __init__(self, model, dataloader, device,):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.scale = conf['data']['train']['scale']
        self.use_Y = conf['trainer']['use_Y_channel_in_val']
        self.psnr_metric = MeanPSNR().to(device)

    def mean_psnr(self):
        self.model.eval()
        psnr_metric = self.psnr_metric
        with torch.no_grad():
            for data in tqdm(self.dataloader, desc="Evaluating PSNR"):
                lr_images = data['img_lr'].to(self.device)
                hr_images = data['img_hr'].to(self.device)

                lr_images = lr_images.type(torch.float32) / 255.
                hr_images = hr_images.type(torch.float32) / 255.

                lr_images = (lr_images - 0.5) / 0.5
                sr_images = self.model(lr_images)
                sr_images = sr_images / 2 + 0.5

                #######for showing SR, debugging ##########是伐
                # while True:
                #     sr_img = tensor2img(sr_images)
                #     cv2.imshow("Super-Resolved Image", sr_img)
                #     key = cv2.waitKey(0) & 0xFF
                #     # print(key)  # Debug: 打印按键的值
                #     if key == ord('q'):
                #         break
                # cv2.destroyAllWindows()
                ###################

                # Apply transformations before updating metrics
                shaved_sr = shave(sr_images, self.scale)
                shaved_hr = shave(hr_images, self.scale)
                if self.use_Y:
                    shaved_sr = rgb_to_ycbcr(shaved_sr)
                    shaved_hr = rgb_to_ycbcr(shaved_hr)

                psnr_metric.update(torch.clamp(shaved_sr, 0, 1),
                                   torch.clamp(shaved_hr, 0, 1))

        return psnr_metric.compute().item()


class MySSIM:
    def __init__(self, model, dataloader, device, conf):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.conf = conf
        self.scale = conf['data']['train']['scale']
        self.use_Y = conf['trainer']['use_Y_channel_in_val']
        self.ssim_metric = MeanSSIM(channel_num=1 if self.use_Y else 3, data_range=1.0).to(device)

    def mean_ssim(self):
        self.model.eval()
        ssim_metric = self.ssim_metric
        with torch.no_grad():
            for data in tqdm(self.dataloader, desc="Evaluating SSIM"):
                lr_images = data['img_lr'].to(self.device)
                hr_images = data['img_hr'].to(self.device)

                lr_images = lr_images.type(torch.float32) / 255.
                hr_images = hr_images.type(torch.float32) / 255.

                lr_images = (lr_images - 0.5) / 0.5
                sr_images = self.model(lr_images)
                sr_images = sr_images / 2 + 0.5

                # Apply transformations before updating metrics
                shaved_sr = shave(sr_images, self.scale)
                shaved_hr = shave(hr_images, self.scale)
                if self.use_Y:
                    shaved_sr = rgb_to_ycbcr(shaved_sr)
                    shaved_hr = rgb_to_ycbcr(shaved_hr)

                ssim_metric.update(torch.clamp(shaved_sr, 0, 1),
                                   torch.clamp(shaved_hr, 0, 1))

        return ssim_metric.compute().item()


def tensor2img(x):
    # Convert the tensor to a NumPy array for displaying
    img = x.squeeze().cpu().numpy().transpose(1, 2, 0)  # Assuming the tensor is (C, H, W)
    img = np.clip(img, 0, 1)  # Ensure the values are in the [0, 1] range
    img = (img * 255).astype('uint8')  # Scale to 0-255 for display
    # Convert from RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

if __name__ == '__main__':
    conf = conf_utils.get_config(path='configs/conf.yaml')
    SR_model = 'standard' # "standard" or "mini", ## setup conf.yaml accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if SR_model == 'standard':
        model = trainer.SRTrainer(conf).to(device)
        pretrained_path = './logs/ckpts/RTSR_N16_epochs180.pth'
        model.network.load_state_dict(torch.load(pretrained_path), strict=conf['strict_load'])
        print(f'{pretrained_path} loaded!')
    if SR_model == 'mini':
        model = trainer.SRTrainer(conf).to(device)
        pretrained_path = './logs/ckpts/RTSR_N8_epochs600.pth'
        model.network.load_state_dict(torch.load(pretrained_path), strict=conf['strict_load'])
        print(f'{pretrained_path} loaded!')

    else:
        print('Model shall be one of: "standard" or "mini"!')


    ds = dataset.SRDataset(conf_data=conf['data']['test'])
    loader = torch.utils.data.DataLoader(dataset=ds, **conf['loader']['test'])

    psnr_evaluator = MyPSNR(model=model, dataloader=loader, device=device)
    ssim_evaluator = MySSIM(model=model, dataloader=loader, device=device, conf=conf)

    mean_psnr_value = psnr_evaluator.mean_psnr()
    mean_ssim_value = ssim_evaluator.mean_ssim()

    print(f'Mean PSNR: {mean_psnr_value}')
    print(f'Mean SSIM: {mean_ssim_value}')
