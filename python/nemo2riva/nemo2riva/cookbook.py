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
                logging.error("Your NeMo model class ({}) is not Exportable.".format(obj.cfg.target))
                sys.exit(1)

            try:
                need_autocast = False
                if torch.cuda.is_available:
                    obj = obj.cuda()
                    if cfg.autocast:
                        need_autocast = True
                    if cfg.args.autocast is not None:
                        need_autocast = cfg.args.autocast
                if need_autocast:
                    autocast = torch.cuda.amp.autocast
                else:
                    autocast = nullcontext
                with autocast():
                    logging.info(f"Exporting model with autocast={need_autocast}")
                    in_args = {}
                    if cfg.args.max_batch is not None:
                        in_args["max_batch"] = cfg.args.max_batch
                    if cfg.args.max_dim is not None:
                        in_args["max_dim"] = cfg.args.max_dim
                    # `_get_input_example()` method was introduced in NeMo v1.3.0. For NeMo versions
                    # <1.3.0 `input_module.input_example()` should be used.
                    if hasattr(obj, '_get_input_example'):
                        input_example = obj._get_input_example(**in_args)
                    else:
                        input_example = obj.input_module.input_example(**in_args)

                    _, descriptions = obj.export(
                        export_file,
                        input_example=input_example,
                        check_trace=cfg.args.runtime_check,
                        onnx_opset_version=cfg.args.onnx_opset,
                        verbose=cfg.args.verbose,
                    )
            except Exception as e:
                logging.error(
                    "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
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
