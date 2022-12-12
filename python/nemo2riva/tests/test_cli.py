#!/usr/bin/env python3
# Copyright (c) 2021-22, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import json
import os
import tarfile
import tempfile

import pytest
from nemo2riva.cli.nemo2riva import nemo2riva
from nemo.collections.nlp.models.machine_translation import MTEncDecModel
from nemo.collections.tts.models import HifiGanModel
from nemo.utils.app_state import AppState

# super simple sanity checks
default_expected_content = ["manifest.yaml", "model_config.yaml"]
onnx_expected_content = default_expected_content + ["model_graph.onnx"]

g_model = HifiGanModel.from_pretrained(model_name="tts_hifigan")


def get_nemo_file(model=g_model):
    model_metadata = AppState().get_model_metadata_from_guid(model.model_guid)
    nemo_file = model_metadata.restoration_path
    return nemo_file


def check_tar_names(riva, content):
    with tarfile.open(riva, "r:gz") as tar:
        tar_names = tar.getnames()
        print("contents: ", tar_names)
        # Everything included in the tarfile
        for expected in content:
            assert os.path.join('artifacts', expected) in tar_names or expected in tar_names


def test_onnx_with_check():
    with tempfile.TemporaryDirectory() as restore_folder:
        riva = os.path.join(restore_folder, "test.riva")
        argv = [get_nemo_file(), "--out", riva, "--runtime-check", "--key", "TEST_KEY"]
        nemo2riva(argv)
        check_tar_names(riva, onnx_expected_content)


def test_nmt_multi_onnx():
    model = MTEncDecModel.from_pretrained("mnmt_deesfr_en_transformer12x2")
    multi_onnx_expected_content = default_expected_content + [
        "encoder_model_graph.onnx",
        "decoder_model_graph.onnx",
        "log_softmax_model_graph.onnx",
        "decoder_tokenizer.model",
        "encoder_tokenizer.model",
    ]
    with tempfile.TemporaryDirectory() as restore_folder:
        riva = os.path.join(restore_folder, "nmt_test.riva")
        argv = [get_nemo_file(model), "--out", riva, "--key", "TEST_KEY"]
        nemo2riva(argv)
        check_tar_names(riva, multi_onnx_expected_content)


if __name__ == '__main__':
    test_ts_override()
