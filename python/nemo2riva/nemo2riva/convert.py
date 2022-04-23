# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import logging
import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from nemo2riva.artifacts import get_artifacts
from nemo2riva.cookbook import save_archive
from nemo2riva.schema import get_export_config, validate_archive
from nemo.core import ModelPT


def Nemo2Riva(args):
    """Convert a .nemo saved model into .riva Riva input format."""
    nemo_in = args.source
    riva_out = args.out

    if riva_out is None:
        riva_out = nemo_in

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    try:
        # Restore instance from .nemo file using generic model restore_from
        model = ModelPT.restore_from(restore_path=nemo_in)
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e

    logging.info("Model {} restored from '{}'".format(model.cfg.target, nemo_in))

    cfg = get_export_config(model, args)

    # Change postfix.
    riva_out = os.path.splitext(riva_out)[0] + ".riva"

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

    patch_kwargs = {}
    if args.export_subnet:
        patch_kwargs['export_subnet'] = args.export_subnet
    model.eval()
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            artifacts, manifest = get_artifacts(restore_path=nemo_in, model=model, passphrase=key, **patch_kwargs)
            if args.export_subnet:
                model = getattr(model, args.export_subnet, None)
                riva_out = riva_out.split(".")
                riva_out[-2] += "-" + args.export_subnet
                riva_out = (".").join(riva_out)
                if model is None:
                    logging.error("Failed to find subnetwork named: {}.".format(args.export_subnet))
                    sys.exit(1)

            save_archive(model=model, save_path=riva_out, cfg=cfg, artifacts=artifacts, metadata=manifest['metadata'])

    logging.info("Successfully exported model to {} and saved to {}".format(cfg.export_file, riva_out))

    if args.validate:
        validate_archive(riva_out, schema=cfg.validation_schema)
