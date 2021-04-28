import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from NABTraf import NABTraf
from tadgan import TadGAN

if __name__ == '__main__':
    data = NABTraf()
    net = TadGAN(in_size=1)

    trainer = pl.Trainer(plugins=[DDPPlugin(find_unused_parameters=False)], fast_dev_run=True, weights_summary='full',
                         log_gpu_memory=True, gpus=[6, 7])
    trainer.fit(net, datamodule=data)
