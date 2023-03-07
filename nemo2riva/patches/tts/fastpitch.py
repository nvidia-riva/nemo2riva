# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import nemo
import yaml
from packaging.version import Version

def fastpitch_model_versioning(model, artifacts, **kwargs):
    # Riva supports some additional features over NeMo fastpitch models depending on the version
    # Namely, we need to patch in volume support and ragged batched support for lower NeMo versions
    try:
        nemo_version = Version(nemo.__version__)
    except NameError:
        # If can't find the nemo version, return without patching
        return None
    if model.__class__.__name__ == 'FastPitchModel':
        # For NeMo version >= 1.11.0; set the relevant flags
        model.export_config["enable_volume"] = True
        model.export_config["enable_ragged_batches"] = True

        # Patch the model config yaml to add the volume and ragged batch flags
        for art in artifacts:
            if art == 'model_config.yaml':
                model_config = yaml.safe_load(artifacts['model_config.yaml']['content'])
                model_config["export_config"] = {'enable_volume': True, 'enable_ragged_batches': True}
                artifacts['model_config.yaml']['content'] = yaml.dump(model_config).encode()
