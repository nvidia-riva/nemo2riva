# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from nemo.core import ModelPT
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from omegaconf import OmegaConf, open_dict
from lightning.pytorch import Trainer


from nemo2riva.artifacts import get_artifacts
from nemo2riva.cookbook import export_model, save_archive
from nemo2riva.schema import get_import_config, get_subnet, validate_archive



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
        accelerator='auto',
        num_nodes=1,
        devices=1,
        # Need to set the following two to False as ExpManager will take care of them differently.
        logger=False,
        enable_checkpointing=False,
    )
    cfg_trainer = OmegaConf.to_container(OmegaConf.create(cfg_trainer))
    trainer = Trainer(**cfg_trainer)

    try:
        with torch.inference_mode():
            if args.load_ckpt:
                if not args.model_config:
                    raise ValueError("Hparams file is required when loading from checkpoint")
                model_cfg = OmegaConf.load(args.model_config)
                ckpt = torch.load(nemo_in, weights_only=False)
                if "state_dict" in ckpt.keys():
                    ckpt = ckpt["state_dict"]

                if "cfg" in model_cfg:
                    model_cfg = model_cfg.cfg
                with open_dict(model_cfg):
                    if model_cfg.target.split(".")[-1] == "MagpieTTSModel":
                        from nemo2riva.patches.tts.magpietts import update_config, update_ckpt
                        from nemo.collections.tts.models.magpietts import MagpieTTSModel
                        legacy_codebooks = False
                        if not args.audio_codecpath:
                            raise ValueError("Audio codec path is required when loading from checkpoint for MagpieTTSModel.")
                        model_cfg = update_config(model_cfg, args.audio_codecpath, legacy_codebooks)
                        state_dict = update_ckpt(ckpt)
                        
                        model = MagpieTTSModel(cfg=model_cfg)
                        model.load_state_dict(state_dict)
                        model.cuda()
                        model.eval()
                        model = model.half()
                    else:
                        model = ModelPT(cfg=model_cfg)
                        model.load_state_dict(ckpt)
            else:
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
            patch_kwargs = {"import_config" : cfg}
            if model.__class__.__name__ == "MagpieTTSModel":
                patch_kwargs['is_encoder'] = args.submodel == "encoder"
            if args.export_subnet:
                patch_kwargs['export_subnet'] = args.export_subnet
            artifacts, manifest = get_artifacts(restore_path=nemo_in, model=model, passphrase=key, model_cfg=args.model_config, from_ckpt=args.load_ckpt, **patch_kwargs)

            for export_cfg in cfg.exports:
                subnet = get_subnet(model, export_cfg.export_subnet)
                export_model(
                    model=subnet, cfg=export_cfg, args=args, artifacts=artifacts, metadata=manifest['metadata']
                )

            save_archive(model=model, save_path=riva_out, cfg=cfg, artifacts=artifacts, metadata=manifest['metadata'])

    logging.info("Model saved to {}".format(riva_out))

    if args.validate:
        validate_archive(riva_out, schema=cfg.validation_schema)
