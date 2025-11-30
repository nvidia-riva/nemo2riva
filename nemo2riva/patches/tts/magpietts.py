# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import random
import nemo
import torch
import yaml
from nemo.core.neural_types.neural_type import NeuralType
from packaging.version import Version
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors


def update_ckpt(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        if 't5_encoder' in key:
            new_key = key.replace('t5_encoder', 'encoder')
            new_state_dict[new_key] = state_dict[key]
        elif 't5_decoder' in key:
            new_key = key.replace('t5_decoder', 'decoder')
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def update_config(model_cfg, codecmodel_path, legacy_codebooks=False):
    ''' helper function to rename older yamls from t5 to magpie '''
    model_cfg.codecmodel_path = codecmodel_path
    if hasattr(model_cfg, 'text_tokenizer'):
        # Backward compatibility for models trained with absolute paths in text_tokenizer
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
    model_cfg.train_ds = None
    model_cfg.validation_ds = None
    if "t5_encoder" in model_cfg:
        model_cfg.encoder = model_cfg.t5_encoder
        del model_cfg.t5_encoder
    if "t5_decoder" in model_cfg:
        model_cfg.decoder = model_cfg.t5_decoder
        del model_cfg.t5_decoder
    if hasattr(model_cfg, 'decoder') and hasattr(model_cfg.decoder, 'prior_eps'):
        # Added to prevent crash after removing arg from transformer_2501.py in https://github.com/blisc/NeMo/pull/56
        del model_cfg.decoder.prior_eps
    if legacy_codebooks:
        # Added to address backward compatibility arising from
        #  https://github.com/blisc/NeMo/pull/64
        print("WARNING: Using legacy codebook indices for backward compatibility. Should only be used with old checkpoints.")
        num_audio_tokens_per_codebook = model_cfg.num_audio_tokens_per_codebook
        model_cfg.forced_num_all_tokens_per_codebook = num_audio_tokens_per_codebook
        model_cfg.forced_audio_eos_id = num_audio_tokens_per_codebook - 1
        model_cfg.forced_audio_bos_id = num_audio_tokens_per_codebook - 2
        if model_cfg.model_type == 'decoder_context_tts':
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 3
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 4
            model_cfg.forced_mask_token_id = num_audio_tokens_per_codebook - 5
        else:
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 1
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 2

    return model_cfg


class EncoderOnnxModel(torch.nn.Module):
    def __init__(self, model, tokenizer_name="english_phoneme"):
        super().__init__()
        model = model.eval().half()
        self.tokenizer_name=tokenizer_name
        self.tokenizer=model.tokenizer
        self.bos_id=model.bos_id
        self.eos_id=model.eos_id
        self.text_embedding=model.text_embedding
        self.encoder=model.encoder


    def forward(self,tokens, token_mask):
        emb_text=self.text_embedding(tokens)
        output=self.encoder(emb_text, token_mask, None, None, None, None, None)
        return output
    
    def _prepare_for_export(self):
        text = "Hello world! How are you doing today?"
        n_batches = 2
        text_encoding = [self.bos_id] + self.tokenizer.encode(text, self.tokenizer_name) + [self.eos_id]
        text_encoding = torch.IntTensor([text_encoding for _ in range(n_batches)]).cuda()
        
        text_lens = torch.IntTensor([text_encoding.shape[1] for _ in range(n_batches)]).cuda()
        max_text_len = torch.max(text_lens).item()
        text_mask = get_mask_from_lengths(text_lens).cuda()  # (B, T)
        
        dummy_output = self(text_encoding, text_mask)

        input_names = ["tokens", "token_mask"]
        output_names = ["output"]
        dynamic_axes = {
            "tokens": {
                0: "batch_size",
                1: "n_texts"
            },
            "token_mask": {
                0: "batch_size",
                1: "n_texts"
            }
        }
        inputs_args = {
            'tokens': text_encoding,
            'token_mask': text_mask,

        }
        return inputs_args, dynamic_axes, output_names, input_names


def magpietts_model_versioning(model, artifacts, **kwargs):
    # Patch for generating magpieTTS model versions
    try:
        nemo_version = Version(nemo.__version__)
    except NameError:
        # If can't find the nemo version, return without patching
        return None

    # Don't override built-in format
    # export_format is read from schemas, radtts is still currently torchscript in the schema
    format_= kwargs['import_config'].exports[0].export_format

    # Patch the model config yaml to add the volume and ragged batch flags
    for art in artifacts:
        from nemo.collections.tts.modules.magpietts_modules import SpecialAudioToken
        if art == 'model_config.yaml':
            model_config = yaml.safe_load(artifacts['model_config.yaml']['content'])["cfg"]
            model_config["target"] = "nemo.collections.tts.models.t5tts.T5TTS_Model"
            if kwargs['is_encoder']:
                model_config["target"] = model_config["target"] + ".text_encoder"
            else:
                model_config['num_audio_tokens_per_codebook'] = model.num_all_tokens_per_codebook # - len(SpecialAudioToken)
                model_config['num_audio_codebooks'] = model.num_audio_codebooks
                model_config["export_config"] = {'enable_ragged_batches': True}
            artifacts['model_config.yaml']['content'] = yaml.dump(model_config).encode()
            