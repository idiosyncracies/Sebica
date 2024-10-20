import os
import torch
import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin #old version to lighting 1.5.0
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import conf_utils
from tools import trainer

conf = conf_utils.get_config(path='configs/conf.yaml')

if __name__ == '__main__':
    pl_module = trainer.SRTrainer(conf)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(  ## make sure val_psnr is assigned in validation_step
        monitor="val_psnr",
        dirpath='./logs/ckpts/',
        filename=conf['name'] + "_{epoch:04d}_{val_psnr:.4f}_{val_ssim:.4f}",
        # filename= 'temp',
        save_top_k=1,
        mode="max",
        # every_n_epochs=1,
    )
    logger = TensorBoardLogger('./logs')
    # callbacks = [lr_monitor, checkpoint_callback]
    callbacks = [lr_monitor] ## 因为checkpoint_callback 保存有BUG 手动保存ckpt

    if conf['load_pretrained']:
        pretrained_path = conf['pretrained_path']
        ext = os.path.splitext(pretrained_path)[1]
        if ext == '.pth':
            pl_module.network.load_state_dict(torch.load(pretrained_path), strict=conf['strict_load'])
            print(f'{pretrained_path} loaded!')
        elif ext == '.ckpt':  ### TODO , Bugs for operating after loading .ckpt file
            # pl_module.load_from_checkpoint(pretrained_path, conf, strict=conf['strict_load'])
            pl_module = trainer.SRTrainer.load_from_checkpoint(pretrained_path, conf)
            print(f'{pretrained_path} loaded!')

    trainer = pl.Trainer(
                         # gpus=[0], #old version
                         devices = [0],
                         accelerator="cuda", # old version ddp
                         # plugins=DDPPlugin(find_unused_parameters=False), # old version
                         strategy=DDPStrategy(),
                         max_epochs=conf['trainer']['num_epochs'],
                         callbacks=callbacks,
                         logger = None,
                         # logger= logger, #None, # old logger = logger,  CHECKPOINT保存后无法LOAD，故移除或简化无法序列化的对象
                         # track_grad_norm=-1, # old version
                         profiler=None,
                         check_val_every_n_epoch=conf['trainer']['check_val_every_n_epoch'],
                         # replace_sampler_ddp=True) # old version
                        )

    trainer.fit(pl_module)
    ## save weight
    # psnr, ssim =0.0,0.0
    psnr = trainer.callback_metrics['best_val_psnr'].detach().item()
    ssim = trainer.callback_metrics['best_val_ssim'].detach().item()
    epochs = conf['trainer']['num_epochs']
    N = conf['network']['params']['N']

    torch.save(pl_module.network.state_dict(), f'./logs/ckpts/RTSR_N{N}_epochs{epochs}_psnr{psnr:.4f}_ssim{ssim:.4f}.pth')