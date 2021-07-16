# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import subprocess
from distutils import cmd as distutils_cmd
from distutils import log as distutils_log
from pathlib import Path

from setuptools import Extension, setup


install_requirements = ["isort<5.0", "nemo_toolkit>=1.0.0", "nvidia-eff>=0.2.9"]

packages = [
    "nemo2riva",
    "nemo2riva.cli",
]

setup_py_dir = Path(__file__).parent.absolute()


def get_version():
    version_file = setup_py_dir / ".." / ".." / "VERSION"
    versions = open(version_file, "r").readlines()
    version = "devel"
    for v in versions:
        if v.startswith("RIVA_VERSION: "):
            version = v[len("RIVA_VERSION: ") :].strip()
    return version


###############################################################################
#                            Code style checkers                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


class StyleCommand(distutils_cmd.Command):
    __LINE_WIDTH = 119
    __ISORT_BASE = (
        "isort "
        # These two lines makes isort compatible with black.
        "--multi-line=3 --trailing-comma --force-grid-wrap=0 "
        f"--use-parentheses --line-width={__LINE_WIDTH} -rc -ws"
    )
    __BLACK_BASE = f"black --skip-string-normalization --line-length={__LINE_WIDTH}"
    description = "Checks overall project code style."
    user_options = [
        ("scope=", None, "Folder of file to operate within."),
        ("fix", None, "True if tries to fix issues in-place."),
    ]

    def __call_checker(self, base_command, scope, check):
        command = list(base_command)

        command.append(scope)

        if check:
            command.extend(["--check", "--diff"])

        self.announce(
            msg="Running command: %s" % str(" ".join(command)), level=distutils_log.INFO,
        )

        return_code = subprocess.call(command)

        return return_code

    def _isort(self, scope, check):
        return self.__call_checker(base_command=self.__ISORT_BASE.split(), scope=scope, check=check,)

    def _black(self, scope, check):
        return self.__call_checker(base_command=self.__BLACK_BASE.split(), scope=scope, check=check,)

    def _pass(self):
        self.announce(msg="\033[32mPASS\x1b[0m", level=distutils_log.INFO)

    def _fail(self):
        self.announce(msg="\033[31mFAIL\x1b[0m", level=distutils_log.INFO)

    # noinspection PyAttributeOutsideInit
    def initialize_options(self):
        self.scope = "."
        self.fix = ""

    def run(self):
        scope, check = self.scope, not self.fix
        isort_return = self._isort(scope=scope, check=check)
        black_return = self._black(scope=scope, check=check)

        if isort_return == 0 and black_return == 0:
            self._pass()
        else:
            self._fail()
            exit(isort_return if isort_return != 0 else black_return)

    def finalize_options(self):
        pass


setup(
    description="NeMo Model => Riva Deployment Converter",
    author="NVIDIA",
    author_email="nvidia.com",
    version=get_version(),
    setup_requires="nvidia-pyindex",
    install_requires=install_requirements,
    packages=packages,
    name="nemo2riva",
    python_requires=">=3.6.0",
    include_package_data=True,
    package_dir={"nemo2riva": "nemo2riva"},
    package_data={"nemo2riva": ["validation_schemas/*.yaml"]},
    entry_points={"console_scripts": ["nemo2riva = nemo2riva.cli:nemo2riva",]},
    cmdclass={"style": StyleCommand},
)
