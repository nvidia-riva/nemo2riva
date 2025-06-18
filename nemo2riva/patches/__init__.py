# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from nemo2riva.patches.ctc import set_decoder_num_classes
from nemo2riva.patches.ctc_bpe import bpe_check_inputs_and_version
from nemo2riva.patches.aed_canary import config_for_trtllm
from nemo2riva.patches.mtencdec import change_tokenizer_names
from nemo2riva.patches.tts import fastpitch_model_versioning, generate_vocab_mapping, radtts_model_versioning

patches = {
    'EncDecCTCModel': {
        'default': [set_decoder_num_classes],
        'onnx': [set_decoder_num_classes],
    },
    'EncDecCTCModelBPE': {
        'default': [bpe_check_inputs_and_version],
        'onnx': [bpe_check_inputs_and_version],
    },
    'EncDecMultiTaskModel': {
        'default': [config_for_trtllm],
        'nemo': [],
    },
    'MTEncDecModel': {
        'default': [change_tokenizer_names],
        'onnx': [change_tokenizer_names],
    },
    'FastPitchModel': {
        'default': [generate_vocab_mapping, fastpitch_model_versioning],
        'onnx': [generate_vocab_mapping, fastpitch_model_versioning],
    },
    'RadTTSModel': {
        'default': [generate_vocab_mapping, radtts_model_versioning],
        'onnx': [generate_vocab_mapping, radtts_model_versioning],
    },
}
