# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from nemo2riva.patches.ctc import set_decoder_num_classes
from nemo2riva.patches.ctc_bpe import bpe_check_inputs_and_version
from nemo2riva.patches.mtencdec import change_tokenizer_names
from nemo2riva.patches.tts import fastpitch_model_versioning, generate_vocab_mapping, radtts_model_versioning

patches = {
    "EncDecCTCModel": [set_decoder_num_classes],
    "EncDecCTCModelBPE": [bpe_check_inputs_and_version],
    "MTEncDecModel": [change_tokenizer_names],
    "FastPitchModel": [generate_vocab_mapping, fastpitch_model_versioning],
    "RadTTSModel": [generate_vocab_mapping, radtts_model_versioning],
}
