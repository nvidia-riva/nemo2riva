# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging
import os
import tarfile
import traceback
from typing import Optional

# from nemo.core.config import hydra_runner
# from nemo.core import ModelPT
from nemo.utils import model_utils


def art_from_name(tar, art_path, conf_name, description=""):
    try:
        member = tar.getmember(art_path)
        if member.isfile():
            f = tar.extractfile(member)
            artifact_content = f.read()
            # Use file name as key (not the whole path).
            _, file_key = os.path.split(member.name)

            # Add artifact.
            return (
                file_key,
                {
                    "description": description,
                    "conf_path": conf_name,
                    "path_type": model_utils.ArtifactPathType.TAR_PATH,
                    "nemo_artifact": True,
                    "content": artifact_content,
                },
            )
    except Exception:
        tb = traceback.format_exc()
        logging.error(f"Could not retrieve the artifact {art_path} used in {conf_name}. Error occured:\n{tb}")
    return None, None


def retrieve_artifacts_as_dict(restore_path: str, obj: Optional["ModelPT"] = None, binary: bool = False):
    """Retrieves all NeMo artifacts and returns them as dict.
        If model object is passed, it first copies artifacts that it "stores" internally.

        Args:
            restore_path: path to file from which the model was restored (and which contains the artifacts).
            obj: ModelPT object (Optional, DEFAULT: None)
            binary: Flag indicating that we want to read/return the content of the files in the binary format (DEFAULT: False).

        Returns:
            Dictionary addressed by filenames. Depending on binary content of files can be binary or string.
        """
    # Returned dictionary.
    artifacts = {}

    # Set read mode depending on the format (string/binary) - for all files.
    read_mode = "rb" if binary else "r"

    # Open the archive.
    with tarfile.open(restore_path, "r:gz") as tar:
        tar_names = tar.getnames()
        # Everything included in the tarfile
        for name in tar_names:
            key, content = art_from_name(tar, name, name)
            if (
                key is not None
                and not key.endswith(".ckpt")
                and not key.endswith(".pt")
                and not key.endswith(".ts")
                and not key.endswith("~")
            ):
                artifacts[key] = content

    logging.info(f"Retrieved artifacts: {artifacts.keys()}")
    return artifacts
