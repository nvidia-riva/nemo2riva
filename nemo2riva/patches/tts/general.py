# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from typing import Optional, Tuple

import torch

check_ipa_support = True
try:
    from nemo.collections.tts.torch.tts_tokenizers import IPATokenizer
except Exception:
    logging.info("IPATokenizer not found in NeMo, disabling support")
    check_ipa_support = False


@torch.jit.script
def create_batch(
    text: torch.Tensor,
    pitch: torch.Tensor,
    pace: torch.Tensor,
    batch_lengths: torch.Tensor,
    padding_idx: int = -1,
    volume: Optional[torch.Tensor] = None,
):
    batch_lengths = batch_lengths.to(torch.int64)
    max_len = torch.max(batch_lengths[1:] - batch_lengths[:-1])

    index = 1
    texts = torch.zeros(batch_lengths.shape[0] - 1, max_len, dtype=torch.int64, device=text.device) + padding_idx
    pitches = torch.zeros(batch_lengths.shape[0] - 1, max_len, dtype=torch.float32, device=text.device)
    paces = torch.zeros(batch_lengths.shape[0] - 1, max_len, dtype=torch.float32, device=text.device) + 1.0
    volumes = torch.zeros(batch_lengths.shape[0] - 1, max_len, dtype=torch.float32, device=text.device) + 1.0

    while index < batch_lengths.shape[0]:
        seq_start = batch_lengths[index - 1]
        seq_end = batch_lengths[index]
        cur_seq_len = seq_end - seq_start

        texts[index - 1, :cur_seq_len] = text[seq_start:seq_end]
        pitches[index - 1, :cur_seq_len] = pitch[seq_start:seq_end]
        paces[index - 1, :cur_seq_len] = pace[seq_start:seq_end]
        if volume is not None:
            volumes[index - 1, :cur_seq_len] = volume[seq_start:seq_end]

        index += 1

    return texts, pitches, paces, volumes

@torch.jit.script
def batch_from_ragged(
    text: torch.Tensor,
    pitch: torch.Tensor,
    pace: torch.Tensor,
    batch_lengths: torch.Tensor,
    padding_idx: int = -1,
    volume: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Same function as create_batch, but updated in NeMo #6020 for 1.17.0
    """

    batch_lengths = batch_lengths.to(dtype=torch.int64)
    max_len = torch.max(batch_lengths[1:] - batch_lengths[:-1])

    index = 1
    num_batches = batch_lengths.shape[0] - 1
    texts = torch.zeros(num_batches, max_len, dtype=torch.int64, device=text.device) + padding_idx
    pitches = torch.ones(num_batches, max_len, dtype=torch.float32, device=text.device)
    paces = torch.zeros(num_batches, max_len, dtype=torch.float32, device=text.device) + 1.0
    volumes = torch.zeros(num_batches, max_len, dtype=torch.float32, device=text.device) + 1.0
    lens = torch.zeros(num_batches, dtype=torch.int64, device=text.device)
    last_index = index - 1
    while index < batch_lengths.shape[0]:
        seq_start = batch_lengths[last_index]
        seq_end = batch_lengths[index]
        cur_seq_len = seq_end - seq_start
        lens[last_index] = cur_seq_len
        texts[last_index, :cur_seq_len] = text[seq_start:seq_end]
        pitches[last_index, :cur_seq_len] = pitch[seq_start:seq_end]
        paces[last_index, :cur_seq_len] = pace[seq_start:seq_end]
        if volume is not None:
            volumes[last_index, :cur_seq_len] = volume[seq_start:seq_end]
        last_index = index
        index += 1

    return texts, pitches, paces, volumes, lens


def generate_vocab_mapping_arpabet(labels):
    mapping = []
    for idx, token in enumerate(labels):
        if not str.islower(token) and str.isalnum(token):
            # token is ARPABET token, need to be prepended with @
            token = '@' + token
        mapping.append("{} {}".format(idx, token))
        if str.islower(token) and str.isalnum(token):
            # normal lowercase token, we want to create uppercase variant too
            # since nemo preprocessing includes a .tolower
            mapping.append("{} {}".format(idx, token.upper()))
    return mapping


def generate_vocab_mapping_ipa(labels):
    # Only support English IPA dict
    VALID_NON_ALNUM_IPA_TOKENS = ['ˈ', 'ˌ', 'ː']
    mapping = []
    for idx, token in enumerate(labels):
        if token in VALID_NON_ALNUM_IPA_TOKENS or (str.isalnum(token) and str.islower(token)):
            # This is a phone
            token = '@' + token
        mapping.append("{} {}".format(idx, token))
    return mapping


def generate_vocab_mapping(model, artifacts, **kwargs):
    # TODO Hack to add labels from FastPitch to .riva since that file is not inside the .nemo
    # Task tracked at https://jirasw.nvidia.com/browse/JARS-1169
    ipa_support = False
    if hasattr(model, "vocab"):
        model_vocab = model.vocab
    elif hasattr(model, "tokenizer"):
        model_vocab = model.tokenizer
    else:
        logging.error("Neither vocab nor tokenizer found!")
        return
    logging.info("Adding mapping.txt to output file")
    if hasattr(model_vocab, "labels"):
        labels = model_vocab.labels
    else:
        labels = model_vocab.tokens
        if check_ipa_support:
            ipa_support = isinstance(model_vocab, IPATokenizer)

    if ipa_support:
        mapping = generate_vocab_mapping_ipa(labels)
    else:
        mapping = generate_vocab_mapping_arpabet(labels)

    mapping_txt = "\n".join(mapping).encode('utf-8')

    content = {
        "description": "mapping file",
        "path_type": "TAR_PATH",
        "nemo_artifact": True,
        "content": mapping_txt,
    }
    artifacts["mapping.txt"] = content
