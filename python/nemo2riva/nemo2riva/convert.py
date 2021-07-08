# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import logging
import os
import traceback
import warnings
from dataclasses import dataclass
from typing import Optional

from nemo2riva.artifacts import retrieve_artifacts_as_dict
from nemo2riva.cookbook import Nemo2RivaCookbook
from nemo2riva.schema import get_export_config, validate_archive

# from nemo.core.config import hydra_runner
from nemo.core import ModelPT
from nemo.utils import model_utils


def Nemo2Riva(args):
    """Convert a .nemo saved model into .ejrvs Riva input format."""
    nemo_in = args.source
    ejrvs_out = args.out

    if ejrvs_out is None:
        ejrvs_out = nemo_in

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    # Restore instance from .nemo file using generic model restore_from
    model = ModelPT.restore_from(restore_path=nemo_in)
    logging.info("Model {} restored from '{}'".format(model.cfg.target, nemo_in))

    cfg = get_export_config(model, args)

    # Change postfix.
    ejrvs_out = os.path.splitext(ejrvs_out)[0] + ".ejrvs"

    cb = Nemo2RivaCookbook()

    # Set the same encryption key - in both archives.
    if args.key is not None:
        try:
            with open(args.key, read_mode) as f:
                key = f.read()
        except Exception:
            # literal key
            key = args.key
        cb.set_encryption_key(key)
    elif cfg.should_encrypt:
        logging.warning('Schema says encryption should be used, but no encryption key passed!')

    # Copy artifacts - first retrieve...
    artifacts = retrieve_artifacts_as_dict(obj=model, restore_path=nemo_in, binary=True)

    # TODO Hack to add labels from FastPitch to .ejrvs since that file is not inside the .nemo
    # Task tracked at https://jirasw.nvidia.com/browse/JARS-1169
    if model.__class__.__name__ == 'FastPitchModel' and hasattr(model, 'vocab'):
        logging.info("Adding mapping.txt for FastPitchModel instance to output file")
        labels = model.vocab.labels
        mapping = []
        for idx, token in enumerate(labels):
            if not str.islower(token) and str.isalnum(token):
                # token is ARPABET token, need to be prepended with @
                token = '@' + token
            mapping.append("{} {}".format(idx, token))
        mapping_txt = "\n".join(mapping)

        content = {
            "description": "mapping file for FastPitch",
            "conf_path": "./mapping.txt",
            "path_type": model_utils.ArtifactPathType.TAR_PATH,
            "nemo_artifact": True,
            "content": mapping_txt,
        }
        artifacts["mapping.txt"] = content

    # ... and next add to "output" cookbook.
    for k, v in artifacts.items():
        cb.add_class_file_content(name=k, **v)

    logging.info(
        "{}: converting {} to  {} using {} export format".format(__name__, nemo_in, ejrvs_out, cfg.export_format)
    )

    # Create Archive using the recipe.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        cb.save(obj=model, save_path=ejrvs_out, cfg=cfg)

    logging.info("Successfully exported model to {} and saved to {}".format(cfg.export_file, ejrvs_out))

    if args.validate:
        validate_archive(save_path, schema=cfg.validation_schema)
