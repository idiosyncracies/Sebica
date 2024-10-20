import torch
from models.arch import RTSR as standard_arch
from models.mini_arch import RTSR as  mini_arch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
import os
import time
import numpy as np

class SuperResolution:
    def __init__(self, net,  test_dataloader, device):
        self.test_dataloader = test_dataloader
        self.device = device
        self.net = net


    def predict_and_display(self, start_idx=0):
        latency_list = []
        self.net.eval()
        with torch.no_grad():
            for idx, (x, lbl) in enumerate(tqdm(self.test_dataloader, desc=f'[Test]', smoothing=1.0)):
                if idx < start_idx:
                    continue

                if device == 'cpu':
                    start = time.time()
                else: #cuda
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()  #cuda

                org_x = x.clone()
                y = self.net(x.to(self.device))


                if device == 'cpu':
                    duration = (time.time() - start) *1000 # metrics in ms
                else:  # gpu
                    ender.record()
                    torch.cuda.synchronize()  ###
                    duration = starter.elapsed_time(ender) # metrics in ms
                latency_list.append(duration)

                org_img = tensor2img(org_x)
                sr_img = tensor2img(y)
                cv2.imshow("Origianl Image", org_img)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    continue

                # Display the image
                cv2.imshow("Super-Resolved Image", sr_img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    continue

        cv2.destroyAllWindows()
        return np.mean(latency_list[10:])

def tensor2img(x):
    # Convert the tensor to a NumPy array for displaying
    img = x.squeeze().cpu().numpy().transpose(1, 2, 0)  # Assuming the tensor is (C, H, W)
    img = np.clip(img, 0, 1)  # Ensure the values are in the [0, 1] range
    img = (img * 255).astype('uint8')  # Scale to 0-255 for display
    # Convert from RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

class TestDataset(Dataset):
    def __init__(self, img_path, lbl_path, crop_size=None, scale_factor=1):
        if not isinstance(img_path, (str, bytes, os.PathLike)) or not isinstance(lbl_path, (str, bytes, os.PathLike)):
            raise TypeError("img_path and lbl_path must be a string, bytes, or os.PathLike, not tuple or other types.")

        self.img_names = sorted([name for name in os.listdir(img_path) if name.endswith(('png', 'jpg', 'jpeg'))])
        self.lbl_names = sorted([name for name in os.listdir(lbl_path) if name.endswith(('png', 'jpg', 'jpeg'))])

        if len(self.img_names) != len(self.lbl_names):
            raise ValueError("Mismatch between number of images and labels.")

        self.img_path = img_path
        self.lbl_path = lbl_path

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = self.tensor(Image.open(os.path.join(self.img_path, self.img_names[idx])).convert('RGB'))
        lbl = self.tensor(Image.open(os.path.join(self.lbl_path, self.lbl_names[idx])).convert('RGB'))

        # Crop if crop_size is specified
        if self.crop_size is not None:
            img = transforms.CenterCrop(self.crop_size)(img)
            lbl = transforms.CenterCrop(self.scale_factor * self.crop_size)(lbl)

        return img, lbl

def test_loader(dataset, batch_size=1, num_workers=4):
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=SequentialSampler(dataset),
                            pin_memory=True)
    return dataloader
if __name__ == '__main__':
    arch = 'standard' ## "mini"', 'standard'
    print(f'Super resolve with {arch} model.')

    img_path =os.path.expanduser("~/Documents/datasets/div2k/own/valid/LR")
    lbl_path=os.path.expanduser("~/Documents/datasets/div2k/own/valid/HR")

    # img_path = os.path.join(os.environ['HOME'],"./Documents/datasets/Flickr2K/own/valid/LR")
    # lbl_path = os.path.join(os.environ['HOME'],"./Documents/datasets/Flickr2K/own/valid/HR")


    test_dataset = TestDataset(img_path, lbl_path)
    test_loader = test_loader(test_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arch == 'standard':
        net_ori = standard_arch(sr_rate=4, N=16).to(device) ### for mini N=8
        ckpt ='./logs/ckpts/RTSR_N16_epochs180.pth'

        net_ori.load_state_dict((torch.load(ckpt)))
        sr = SuperResolution(net_ori, test_loader, device)

    elif arch == 'mini':
        net_ori = mini_arch(sr_rate=4, N=8).to(device)  ### for mini N=8
        ckpt = './logs/ckpts/RTSR_N8_epochs600.pth'
    else:
        print('Model shall be one of: "standard" or "mini"!')

    net_ori.load_state_dict((torch.load(ckpt)))
    sr = SuperResolution(net_ori, test_loader, device)


    # sr.predict_and_display()
    latency = sr.predict_and_display()
    print(f'Average fps:: {1000/latency:.4f} .')
