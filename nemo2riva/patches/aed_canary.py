# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import yaml
import json
import logging


def config_for_trtllm(model, artifacts, **kwargs):
    if model.__class__.__name__ == 'EncDecMultiTaskModel':

        model_config = yaml.safe_load(artifacts['model_config.yaml']['content'])

        keys_required = [
            'beam_search',
            'encoder',
            'head',
            'model_defaults',
            'prompt_format',
            'sample_rate',
            'target',
            'preprocessor',
        ]
        if 'beam_search' not in model_config and 'decoding' in model_config:
            model_config['beam_search'] = model_config['decoding'].get('beam', {'beam_size': 1, 'len_pen': 0.0,
                                                                                'max_generation_delta': 50}
                                                                       )
        config = dict({k: model_config[k] for k in keys_required})
        config['decoder'] = {
            'transf_decoder': model_config['transf_decoder'],
            'transf_encoder': model_config['transf_encoder'],
            'vocabulary': make_vocabulary_file(model,artifacts),
            'num_classes': model_config['head']['num_classes'],
            'feat_in': model_config['model_defaults']['asr_enc_hidden'],
            'n_layers': model_config['transf_decoder']['config_dict']['num_layers'],
        }
        config['target'] = 'trtllm.canary'


        artifacts['model_config.yaml']['content'] = yaml.safe_dump(config, encoding=('utf-8'))


def make_vocabulary_file(model, artifacts, **kwargs):
    if model.__class__.__name__ == 'EncDecMultiTaskModel':

        tokenizer_vocab = {'tokens': {},
                           'offsets': model.tokenizer.token_id_offset
                           }
        for lang in model.tokenizer.langs:
            tokenizer_vocab['tokens'][lang] = {}
        tokenizer_vocab['size'] = model.tokenizer.vocab_size

        try:
            tokenizer_vocab['bos_id'] = model.tokenizer.bos_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing bos_id. Could affect accuracy")

        try:
            tokenizer_vocab['eos_id'] = model.tokenizer.eos_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing eos_id. Could affect accuracy")
        try:
            tokenizer_vocab['nospeech_id'] = model.tokenizer.nospeech_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing nospeech_id. Could affect accuracy")
        try:
            tokenizer_vocab['pad_id'] = model.tokenizer.pad_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing pad_id. Could affect accuracy")

        for t_id in range(0, model.tokenizer.vocab_size):
            lang = model.tokenizer.ids_to_lang([t_id])
            tokenizer_vocab['tokens'][lang][t_id] = model.tokenizer.ids_to_tokens([t_id])[0]

        artifacts['vocab.json']={}
        artifacts['vocab.json']['content'] = json.dumps(tokenizer_vocab).encode('utf-8')
        return tokenizer_vocab

