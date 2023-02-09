# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import random
import nemo
import torch
import yaml
from nemo.core.neural_types.neural_type import NeuralType
from packaging.version import Version

from nemo2riva.patches.tts.general import create_batch

def radtts_model_versioning(model, artifacts, **kwargs):
    # We need to patch in
    # - ragged batching support. Not supported with torch backend, tracked as DLIS-4332
    try:
        nemo_version = Version(nemo.__version__)
    except NameError:
        # If can't find the nemo version, return without patching
        return None
    if model.__class__.__name__ == 'RadTTSModel':
        # Patch the model config yaml to add the volume and ragged batch flags
        for art in artifacts:
            if art == 'model_config.yaml':
                model_config = yaml.safe_load(artifacts['model_config.yaml']['content'])
                model_config["export_config"] = {'enable_volume': True, 'enable_ragged_batches': False}
                artifacts['model_config.yaml']['content'] = yaml.dump(model_config).encode()
