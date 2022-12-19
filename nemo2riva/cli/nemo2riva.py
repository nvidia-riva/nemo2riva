# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import sys

from eff.package_info import __version__ as eff_version
from nemo2riva.args import get_args
from nemo2riva.convert import Nemo2Riva
from nemo2riva.cookbook import CudaOOMInExportOfASRWithMaxDim
from nemo.utils import logging

"""

# Exemplary call:
#################

nemo2riva model.nemo

nemo2riva model.nemo --out ../model.riva --format onnx

"""


MINIMUM_ALLOWED_MAX_INPUT_LENGTH_FOR_ASR = 10000


def log_config(args):
    loglevel = logging.INFO
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    if args.verbose is not None:
        numeric_level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % numeric_level)
        loglevel = numeric_level
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(logging.getEffectiveLevel()))


def nemo2riva(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = get_args(argv)
    log_config(args)
    max_dim_is_too_large = True
    while max_dim_is_too_large:
        try:
            Nemo2Riva(args)
            max_dim_is_too_large = False
        except CudaOOMInExportOfASRWithMaxDim as e:
            if e.max_dim <= MINIMUM_ALLOWED_MAX_INPUT_LENGTH_FOR_ASR:
                err_msg = (
                    f"Could not export model {args.source} because of CUDA OOM error even after setting `max_dim` "
                    f"parameter to a value {e.max_dim} which is less or equal to minimum possible "
                    f"value {MINIMUM_ALLOWED_MAX_INPUT_LENGTH_FOR_ASR}."
                )
                logging.error(f"ERROR: {err_msg}")
                raise CudaOOMInExportOfASRWithMaxDim(err_msg)
            else:
                next_max_dim = max(e.max_dim // 2, MINIMUM_ALLOWED_MAX_INPUT_LENGTH_FOR_ASR)
                logging.warning(
                    f"It looks like you're trying to export a ASR model with "
                    f"max_dim={e.max_dim}. Export is failing due to CUDA OOM. Reducing `max_dim` to {next_max_dim} "
                    f"and trying again..."
                )
                args.max_dim = next_max_dim

    eff_base_version = eff_version.split('-')[0]
    if eff_base_version == '0.5.2':
        logger.warning(
            "\n************************************************************************\n"
            "       Please ignore the pyarmor warnings below.\n"
            "       Please upgrade nvidia-eff package to >=0.5.3 if available.\n"
            "************************************************************************\n"
        )


if __name__ == '__main__':
    nemo2riva(sys.argv[1:])
