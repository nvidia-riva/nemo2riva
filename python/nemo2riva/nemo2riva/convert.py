# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from nemo2riva.artifacts import get_artifacts
from nemo2riva.cookbook import export_model, save_archive
from nemo2riva.schema import get_import_config, get_subnet, validate_archive
from nemo.core import ModelPT

from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from pytorch_lightning import Trainer


def Nemo2Riva(args):
    """Convert a .nemo saved model into .riva Riva input format."""
    nemo_in = args.source
    riva_out = args.out
    if riva_out is None:
        riva_out = nemo_in
    # Change postfix.
    riva_out = os.path.splitext(riva_out)[0] + ".riva"
    if args.export_subnet:
        riva_out = riva_out.split(".")
        riva_out[-2] += "-" + args.export_subnet
        riva_out = (".").join(riva_out)

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    # Create a PL trainer object which is required for restoring Megatron models
    cfg_trainer = TrainerConfig(
        gpus=1,
        accelerator="ddp",
        num_nodes=1,
        # Need to set the following two to False as ExpManager will take care of them differently.
        logger=False,
    )
    trainer = Trainer(cfg_trainer)

    try:
        # Restore instance from .nemo file using generic model restore_from
        model = ModelPT.restore_from(restore_path=nemo_in, trainer=trainer)
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e

    cfg = get_import_config(model, args)

    # Set the same encryption key - in both archives.
    key = None
    if args.key is not None:
        try:
            with open(args.key, read_mode) as f:
                key = f.read()
        except Exception:
            # literal key
            key = args.key
    elif cfg.should_encrypt:
        logging.warning('Schema says encryption should be used, but no encryption key passed!')

    model.eval()
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # TODO: revisit export_subnet cli arg
            patch_kwargs = {}
            if args.export_subnet:
                patch_kwargs['export_subnet'] = args.export_subnet
            artifacts, manifest = get_artifacts(restore_path=nemo_in, model=model, passphrase=key, **patch_kwargs)

            for export_cfg in cfg.exports:
                subnet = get_subnet(model, export_cfg.export_subnet)
                export_model(
                    model=subnet, cfg=export_cfg, args=args, artifacts=artifacts, metadata=manifest['metadata']
                )

            save_archive(model=model, save_path=riva_out, cfg=cfg, artifacts=artifacts, metadata=manifest['metadata'])

    logging.info("Model saved to {}".format(riva_out))

    if args.validate:
        validate_archive(riva_out, schema=cfg.validation_schema)
