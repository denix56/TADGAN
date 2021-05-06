import click
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from NABTraf import NABTraf
from tadgan import TadGAN


class IntListType(click.ParamType):
    name = "intlist"

    def convert(self, value, param, ctx):
        try:
            res = value.split(",")
            if len(res) == 1:
                return int(value)
            
            res = [int(val) for val in res if val != '']
            
            return res
        except TypeError:
            self.fail(
                "expected string for list of int() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid list of integers", param, ctx)

            
@click.command()
@click.option('--dataset', type=str, default='nyc_taxi', help='Dataset to use. Check Orion-ML git repo for available datasets')
@click.option('--gpus', type=IntListType(), default=1, help='Specify either number of gpus or their device ids. Add comma at the end if you want to specify the id while using only 1 GPU')
def main(dataset, gpus):
    batch_size = 64
    
    data = NABTraf(batch_size=batch_size, data_path=dataset)
    net = TadGAN(in_size=1, weight_decay=1e-6, iterations_critic=5, lr=0.0005, use_gru=True)
    net.example_input_array = torch.ones(batch_size, 100, 1, dtype=torch.float)
    logger = TensorBoardLogger('logs', name='tadgan', log_graph=True)
    
#     early_stop_callback = EarlyStopping(
#        monitor='F1',
#        min_delta=0.00,
#        patience=3,
#        verbose=True,
#        mode='max'
#     )

    trainer = pl.Trainer(plugins=[DDPPlugin(find_unused_parameters=True)], fast_dev_run=False, weights_summary='full',
                         log_gpu_memory=True, gpus=gpus, accelerator='ddp', logger=logger,
                         check_val_every_n_epoch=5, max_epochs=100, callbacks=[GPUStatsMonitor(), 
                                                                              # early_stop_callback
                                                                              ]
                        )
    trainer.fit(net, datamodule=data)
    
    

if __name__ == '__main__':
    main()
