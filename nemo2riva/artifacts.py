# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import tarfile
import traceback
from typing import Optional

import yaml
from eff.callbacks import (
    BinaryContentCallback,
    ContentCallback,
    PickleContentCallback,
    StringContentCallback,
    VocabularyContentCallback,
    YamlContentCallback,
)
from eff.codec import get_random_encryption
from eff.core import ArtifactRegistry, File, Memory
from nemo.utils import logging

# fmt: off
_HAVE_PATCHES = True
_HAVE_PATCHES_ERROR_MSG = None
try:
    from nemo2riva.patches import patches
except ModuleNotFoundError as e:
    _HAVE_PATCHES = False
    _HAVE_PATCHES_ERROR_MSG = e
# fmt: on


def retrieve_artifacts_as_dict(restore_path: str, obj: Optional["ModelPT"] = None):
    """ Retrieves all NeMo artifacts and returns them as dict
        Args:
            restore_path: path to file from which the model was restored (and which contains the artifacts).
            obj: ModelPT object (Optional, DEFAULT: None)
            binary: Flag indicating that we want to read/return the content of the files in the binary format (DEFAULT: False).

        Returns:
            Dictionary addressed by filenames. Depending on binary content of files can be binary or string.
        """
    # Returned dictionary.
    artifacts = {}

    # Open the archive.
    with tarfile.open(restore_path, "r") as tar:
        tar_names = tar.getnames()
        # Everything included in the tarfile
        for name in tar_names:
            try:
                if (
                    obj is not None
                    and name.endswith(".ckpt")
                    or name.endswith(".pt")
                    or name.endswith(".ts")
                    or name.endswith("~")
                ):
                    logging.info(f"Found model at {name}")
                else:
                    member = tar.getmember(name)
                    _, file_key = os.path.split(member.name)

                    if member.isfile():
                        f = tar.extractfile(member)
                        artifact_content = f.read()
                        aname = member.name
                        if aname.startswith('./'):
                            aname = aname[2:]
                        artifacts[file_key] = {
                            "conf_path": aname,
                            "path_type": "TAR_PATH",
                            "content": artifact_content,
                        }
            except Exception:
                tb = traceback.format_exc()
                logging.error(f"Could not retrieve the artifact {file_key} at {member.name}. Error occured:\n{tb}")
    return artifacts


def create_artifact(reg, key, do_encrypt, **af_dict):
    # only works for plain content now - no encryption in Nemo
    encryption = False
    if 'encrypted' in af_dict:
        do_encrypt = af_dict['encrypted']
        af_dict.pop('encrypted')
    if do_encrypt and key is not None:  # No encryption if no key was given
        encryption = get_random_encryption()

    af = reg.create(name=key, encryption=encryption, **af_dict,)
    if do_encrypt:
        af.encrypt()
    return af


def get_artifacts(restore_path: str, model=None, passphrase=None, **patch_kwargs):
    artifacts = retrieve_artifacts_as_dict(obj=model, restore_path=restore_path)

    # NOTE: when servicemaker calls into get_artifacts, model is always None so this code section
    # is never run.
    # check if this model has one or more patches to apply, if yes go ahead and run it
    if model is not None and _HAVE_PATCHES and model.__class__.__name__ in patches:
        for patch in patches[model.__class__.__name__]:
            patch(model, artifacts, **patch_kwargs)
    elif model is not None and not _HAVE_PATCHES:
        logging.error(
            "nemo2riva's get_artifacts() was called but was unable to continue due to missing "
            "modules. Please ensure that nemo_toolkit and it's dependencies are all installed "
            "before re-running nemo2riva."
        )
        if isinstance(_HAVE_PATCHES_ERROR_MSG, Exception):
            raise _HAVE_PATCHES_ERROR_MSG
        else:
            raise ModuleNotFoundError

    if 'manifest.yaml' in artifacts.keys():
        nemo_manifest = yaml.safe_load(artifacts['manifest.yaml']['content'])
    else:
        nemo_manifest = {'files': artifacts, 'metadata': {'format_version': 1}}
        if 'model_config.yaml' in artifacts.keys():
            nemo_manifest['has_nemo_config'] = True

    nemo_files = nemo_manifest['files']
    nemo_metadata = nemo_manifest['metadata']
    reg = ArtifactRegistry(passphrase=passphrase)

    for key, af_dict in artifacts.items():
        if key in nemo_files:
            af_dict.update(nemo_files[key])
        elif './' + key in nemo_files:
            af_dict.update(nemo_files['./' + key])

        cb_override = BinaryContentCallback
        create_artifact(reg, key, False, content_callback=cb_override, **af_dict)

    logging.info(f"Retrieved artifacts: {artifacts.keys()}")
    return reg, nemo_manifest
