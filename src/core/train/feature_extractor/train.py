from .dataset import ZipPoseDataset, PackedDataset
from ...models import VQ_VAE, AutoEncoderLightningWrapper

import torch
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
import pytorch_lightning as pl

def main():
    feat_enc = VQ_VAE()
    # loss_weights? Read up what this is
    model = AutoEncoderLightningWrapper(feat_enc, learning_rate="lr", loss_weights="loss_weights")

    dataset = ZipPoseDataset("data_path")
    training_dataset = dataset.slice(10, None)
    val_dataset = dataset.slice(0, 10)
    
    training_iter_dataset = PackedDataset(training_dataset)

    train_dataset = DataLoader(
        training_iter_dataset,
        batch_size=12,
        num_workers=1,
        collate_fn=zero_pad_collator
    )
    val_dataset = DataLoader(
        val_dataset,
        batch_size=12,
        shuffle=False,
        num_workers=1,
        collate_fn=zero_pad_collator
    )

    trainer = pl.Trainer(max_steps=5,
        logger="logger",
        callbacks="callbacks",
        val_check_intervall = 100_000 / 12, # batch_size,
        accelerator="device",
        profiler="simple",
        precision="float16",
        gradient_clip_val=1, # same as Llamma 2 
    )

    trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)