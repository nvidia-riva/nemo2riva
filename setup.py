# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import subprocess
from distutils import cmd as distutils_cmd
from distutils import log as distutils_log
from pathlib import Path

from setuptools import setup, find_packages

def req_file(filename):
    with open(Path(filename), encoding='utf-8') as f:
        content = f.readlines()
    return [x.strip() for x in content]

install_requirements = req_file("requirements.txt")

__author_email__ = "nvidia-riva@nvidia.com"
__contact_emails__ = "nvidia-riva@nvidia.com"
__contact_names__ = "NVIDIA Riva"
__description__ = ("NeMo Model => Riva Deployment Converter",)
__license__ = "MIT"
__package_name__ = "nemo2riva"
__version__ = "2.14.0"


setup(
    description=__description__,
    author=__contact_names__,
    author_email=__author_email__,
    version=__version__,
    license=__license__,
    install_requires=install_requirements,
    packages=find_packages(),
    name=__package_name__,
    python_requires=">=3.7.0",
    include_package_data=True,
    package_dir={"nemo2riva": "nemo2riva"},
    package_data={"nemo2riva": ["validation_schemas/*.yaml"]},
    entry_points={"console_scripts": ["nemo2riva = nemo2riva.cli:nemo2riva",]},
)
