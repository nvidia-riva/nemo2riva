#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.tts.models import HifiGanModel
from nemo.utils.app_state import AppState

# super simple sanity checks
default_expected_content = ["manifest.yaml", "model_config.yaml"]
onnx_expected_content = default_expected_content + ["model_graph.onnx"]
ts_expected_content = default_expected_content + ["model_graph.ts"]
ckpt_expected_content = default_expected_content + ["model_weights.ckpt"]

g_model = HifiGanModel.from_pretrained(model_name="tts_hifigan")


def get_nemo_file():
    model_metadata = AppState().get_model_metadata_from_guid(g_model.model_guid)
    nemo_file = model_metadata.restoration_path
    return nemo_file


@pytest.mark.skip("Only one cookbook per process")
def test_onnx_with_check():
    with tempfile.TemporaryDirectory() as restore_folder:
        riva = os.path.join(restore_folder, "test.riva")
        argv = [get_nemo_file(), "--out", riva, "--runtime-check"]
        nemo2riva(argv)
        with tarfile.open(riva, "r:gz") as tar:
            tar_names = tar.getnames()
            # Everything included in the tarfile
            for expected in onnx_expected_content:
                assert expected in tar_names


def test_ts_override():
    with tempfile.TemporaryDirectory() as restore_folder:
        riva = os.path.join(restore_folder, "ts_test.riva")
        argv = [get_nemo_file(), "--out", riva, "--format", "ts"]
        nemo2riva(argv)
        with tarfile.open(riva, "r:gz") as tar:
            tar_names = tar.getnames()
            print("contents: ", tar_names)
            # Everything included in the tarfile
            for expected in ts_expected_content:
                assert expected in tar_names


if __name__ == '__main__':
    test_ts_override()
