# Sebica: Lightweight Spatial and Efficient Bidirectional Channel Attention Super Resolution Network

> We propose a Lightweight Spatial and Efficient Bidirectional Channel Attention Super Resolution Network
> This study is inspired from some similar works, e.g., ais 2024 challenge 
> This study aims to find the optimal network architecture in terms of save computational resource without reducing accuracy. 
> We reserve the further squeeze technichs, e.g. distilation, reparameterize, in the future
> It's support real-time 4K video processing

## Dataset
- Div2K and Flirkr2K. We filter the small sized images, and chopped the images in the unique size of 1152 x 2040 for training and testing
- Use jupyter notebook to process the original data, or:
- Download processed div2K via [here](https://drive.google.com/file/d/1ETSlWtgJvbDZ9nGCXNwtMMSpmMjE2YnP/view?usp=sharing).
- Flirkr hasn't been upload deu to it's huge size, please download in the official website and convert by yourself
- Put the datasets in ~/Documents/Datasets, or others you like, but revise the path in the code accordingly

## Pre-trained weights 
-  It's in the folder of logs/chpts

## Train
- Setup configures in configs/conf.yaml
- Run train.py

## Visualize inferrence result:
- Setup the mode, e.g., "standard" or "mini", and pth file path in the __name__ function
- Run infer.py accordingly

## Evaluate psnr ssim:
- Setup the mode, e.g., "standard" or "mini", and pth file path in the __name__ function of psnr_ssim_evaluate.py
- Setup network, dataset path (in data->test) in conf.yaml accordingly
- Run psnr_ssim_evaluate.py

## Object detection test
- Download the full dataset via official website, or 
- Download the section via [here](https://drive.google.com/file/d/1Se77Pvcll7hVl32LktsV7T-LXmKO7VH3/view?usp=sharing)

## Acknowledgement
Some of this work is based on [Bicubic++](https://github.com/aselsan-research-imaging-team/bicubic-plusplus) and [RVSR](https://github.com/huai-chang/RVSR/tree/main ), thanks to their valuable contributions.

## Citation
- Our study has been published on Arxiv, welcome to cite:
```bibliography
@article{liu2024sebica,
  title={Sebica: Lightweight Spatial and Efficient Bidirectional Channel Attention Super Resolution Network},
  author={Liu, Chongxiao},
  journal={arXiv preprint arXiv:2410.20546},
  year={2024}
}
