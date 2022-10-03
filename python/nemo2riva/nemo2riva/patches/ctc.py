# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import yaml


def set_decoder_num_classes(model, artifacts, **kwargs):
    if model.__class__.__name__ == 'EncDecCTCModel':

        conf = yaml.safe_load(artifacts['model_config.yaml']['content'])

        if ('decoder' in conf) and ('num_classes' in conf['decoder']) and (conf['decoder']['num_classes'] == -1):
            if 'vocabulary' in conf['decoder'] and len(conf['decoder']['vocabulary']) > 0:
                conf['decoder']['num_classes'] = len(conf['decoder']['vocabulary'])

        artifacts['model_config.yaml']['content'] = yaml.safe_dump(conf, encoding=('utf-8'))
