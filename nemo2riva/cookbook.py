# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import gc
import os
import sys
import tarfile
import tempfile

import onnx
import onnx_graphsurgeon as gs
import torch
from eff.callbacks import BinaryContentCallback
from eff.core import Archive, ArtifactRegistry, File
from nemo.core import Exportable, ModelPT
from nemo.utils import logging

from nemo2riva.artifacts import create_artifact

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


class CudaOOMInExportOfASRWithMaxDim(Exception):
    def __init__(self, *args, max_dim=None):
        super().__init__(*args)
        self.max_dim = max_dim


def export_model(model, cfg, args, artifacts, metadata):

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
    # TODO: use submodel sections
    metadata.update(format_meta)
    runtime = format_meta["runtime"]
    metadata.update({"runtime": runtime})

    if cfg.cache_support and hasattr(model, "encoder") and hasattr(model.encoder, "export_cache_support"):
        model.encoder.export_cache_support = True
        logging.info("Caching support is enabled.")

    with tempfile.TemporaryDirectory() as tmpdir:
        export_filename = cfg.export_file
        export_file = os.path.join(tmpdir, export_filename)

        if cfg.export_format in ["ONNX", "TS"]:
            # Export the model, get the descriptions.
            if not isinstance(model, Exportable):
                logging.error("Your NeMo model class ({}) is not Exportable.".format(metadata['obj_cls']))
                sys.exit(1)

            error_msg = (
                "ERROR: Export failed. Please make sure your NeMo model class ({}) has working export() and that "
                "you have the latest NeMo package installed with [all] dependencies.".format(model.__class__)
            )
            try:
                autocast = torch.cuda.amp.autocast(enabled=True, cache_enabled=False, dtype=torch.float16) if cfg.autocast else nullcontext()
                with autocast, torch.no_grad(), torch.inference_mode():
                    logging.info(f"Exporting model {model.__class__.__name__} with config={cfg}")
                    model = model.to(device=args.device)
                    model.freeze()
                    in_args = {}
                    if args.max_batch is not None:
                        in_args["max_batch"] = args.max_batch
                    if cfg.max_dim is not None:
                        in_args["max_dim"] = cfg.max_dim

                    input_example = model.input_module.input_example(**in_args)
                    _, descriptions = model.export(
                        export_file,
                        input_example=input_example,
                        check_trace=args.runtime_check,
                        onnx_opset_version=args.onnx_opset,
                        verbose=args.verbose,
                    )
                    del model
                if cfg.export_format == 'ONNX':
                    o_list = os.listdir(tmpdir)
                    save_as_external_data = len(o_list) > 1
                    # fold-constants part
                    model_onnx = onnx.load_model(export_file)
                    graph = gs.import_onnx(model_onnx)
                    graph.fold_constants().cleanup()
                    model_onnx = gs.export_onnx(graph)
                    # remove bits of original .onnx
                    for f in o_list:
                        os.unlink(os.path.join(tmpdir, f))
                    onnx.save_model(
                        model_onnx,
                        export_file,
                        save_as_external_data=save_as_external_data,
                        all_tensors_to_one_file=False,
                    )
                    del model_onnx
                    if save_as_external_data:
                        o_list = os.listdir(tmpdir)
                        export_file = export_file + '.tar'
                        logging.info(
                            f"Large (>2GB) ONNX is being exported with external weights, as {export_filename} TAR archive!"
                        )
                        with tarfile.open(export_file, "w") as tar:
                            for f in o_list:
                                fpath = os.path.join(tmpdir, f)
                                tar.add(fpath, f)

            except RuntimeError as e:
                if "max_dim" in in_args and "CUDA out of memory" in str(e):
                    raise CudaOOMInExportOfASRWithMaxDim(max_dim=in_args['max_dim'])
                else:
                    logging.error(error_msg)
                    raise e
            except Exception as e:
                logging.error(error_msg)
                raise e

        elif cfg.export_format == "CKPT":
            # Save model state using torch save.
            torch.save(model.state_dict(), export_file)

        # Add exported file to the artifact registry

        create_artifact(
            artifacts,
            export_filename,
            do_encrypt=cfg.encryption,
            filepath=export_file,
            description="Exported model",
            content_callback=BinaryContentCallback,
            **format_meta,
        )


def save_archive(model, save_path, cfg, artifacts, metadata):
    metadata.update(
        {
            "description": "Exported Nemo Model",
            "format_version": 3,
            "has_pytorch_checkpoint": False,
            # use 'normalized' class name
            "obj_cls": cfg.cls,
            "min_nemo_version": "1.3",
        }
    )

    logging.info("Saving to {}".format(save_path))
    # Create EFF archive.
    Archive.save_registry(
        save_path=save_path, registry_name="artifacts", registry=artifacts, **metadata,
    )
    del artifacts
    gc.collect()
