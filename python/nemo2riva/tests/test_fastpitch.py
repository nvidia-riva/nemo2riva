#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import tarfile
from pathlib import Path

import pytest

from nemo2riva.cli.nemo2riva import nemo2riva


def test_nemo_with_labels():
    test_file = os.path.join(os.environ["TEST_FILE_PATH"], "FastPitch_22k_LJS.nemo")
    argv = [test_file]
    nemo2riva(argv)
    # In this case, output will not have mapping.txt file
    with tarfile.open(os.path.join(os.environ["TEST_FILE_PATH"], Path(test_file).stem + ".ejrvs"), "r:gz") as f:
        assert "mapping.txt" not in f.getnames()


def test_nemo_with_phonemes():
    test_file = os.path.join(os.environ["TEST_FILE_PATH"], "FastPitch_44k_8051.nemo")
    argv = [test_file]
    nemo2riva(argv)
    # In this case, output will have mapping.txt file
    with tarfile.open(os.path.join(os.environ["TEST_FILE_PATH"], Path(test_file).stem + ".ejrvs"), "r:gz") as f:
        assert "mapping.txt" in f.getnames()


if __name__ == '__main__':
    pass
