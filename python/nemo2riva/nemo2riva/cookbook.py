import gc
import logging
import os
import sys
import tempfile

import torch
from eff.callbacks import BinaryContentCallback
from eff.core import Archive, ArtifactRegistry, File
from nemo.core import Exportable, ModelPT

from .artifacts import create_artifact

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


def save_archive(obj, save_path, cfg, artifacts, metadata):

    metadata.update(
        {
            "description": "Exported Nemo Model, in {} format.".format(cfg.export_format),
            "format_version": 3,
            "has_pytorch_checkpoint": False,
            # use 'normalized' class name
            "obj_cls": cfg.cls,
            "min_nemo_version": "1.3",
        }
    )

    if cfg.export_format == "TS":
        format_meta = {
            "torchscript": True,
            "nemo_archive_version": 2,
            "runtime": "TorchScript",
        }
    elif cfg.export_format == "ONNX":
        format_meta = {
            "onnx": True,
            "onnx_archive_format": 1,
            "runtime": "ONNX",
        }
    elif cfg.export_format == "CKPT":
        format_meta = {"has_pytorch_checkpoint": True, "runtime": "PyTorch"}
        metadata.update(format_meta)

    runtime = format_meta["runtime"]
    metadata.update({"runtime": runtime})

    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = os.path.join(tmpdir, cfg.export_file)
        if cfg.export_format in ["ONNX", "TS"]:
            # Export the model, get the descriptions.
            if not isinstance(obj, Exportable):
                logging.error("Nemo2Jarvis: Your NeMo model class ({}) is not Exportable.".format(obj.cfg.target))
                sys.exit(1)

            try:
                autocast = nullcontext
                if torch.cuda.is_available:
                    obj = obj.cuda()
                    if cfg.autocast:
                        autocast = torch.cuda.amp.autocast
                with autocast():
                    logging.info(f"Exporting model with autocast={cfg.autocast}")
                    _, descriptions = obj.export(export_file, check_trace=cfg.args.runtime_check)
            except Exception as e:
                logging.error(
                    "Nemo2Jarvis: Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                        obj.cfg.target
                    )
                )
                raise e

        elif cfg.export_format == "CKPT":
            # Save model state using torch save.
            torch.save(obj.state_dict(), export_file)

        # Add exported file to the artifact registry

        create_artifact(
            artifacts,
            cfg.export_file,
            do_encrypt=cfg.encryption,
            filepath=export_file,
            description="Exported model",
            content_callback=BinaryContentCallback,
            **format_meta,
        )

        logging.info("Saving to {}".format(save_path))
        # Create EFF archive.
        Archive.save_registry(
            save_path=save_path, registry_name="artifacts", registry=artifacts, **metadata,
        )
        del artifacts
        gc.collect()
