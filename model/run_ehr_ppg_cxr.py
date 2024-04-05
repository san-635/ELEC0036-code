# run.py for finetune_ehr_ppg_cxr using the windowing approach (w/ or w/o overlap); train, val=test and test
import warnings
import os
import copy
import pytorch_lightning as pl

from vilt.config import ex

from vilt.modules import ViLTransformerSS_win  # adapted transformer model
from vilt.datamodules import EHRPPGCXRDataModule_win  # adapted data loader

warnings.filterwarnings("ignore")

# python run_ehr_ppg_cxr.py with task_finetune_ehr_ppg_cxr load_path=vilt_200k_mlm_itm.ckpt batch_size=6 per_gpu_batchsize=6 gpu_device=
# python run_ehr_ppg_cxr.py with task_finetune_ehr_ppg_cxr batch_size=6 per_gpu_batchsize=6 test_only=True gpu_device= load_path=

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    
    dm = EHRPPGCXRDataModule_win(_config)
    model = ViLTransformerSS_win(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)  # create 'result' dir
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=False,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_{_config["batch_size"]}',
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/the_metric",
        min_delta=0,
        patience=2//_config["val_check_interval"],
        verbose=True,
        mode="max"
    )
    callbacks = [checkpoint_callback, lr_callback, early_stop_callback]
    
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"])  # always 1 in our case
    print(_config["batch_size"], _config["per_gpu_batchsize"], num_gpus, _config["num_nodes"])
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=str(_config["gpu_device"]),
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="cuda",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps, # accumulate gradients after every batch
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)   # trigger the calling of the five steps of the training loop that are defined in MTDataModule
    else:
        trainer.test(model, datamodule=dm)  # trigger the calling of the five steps of the test loop that are defined in MTDataModule