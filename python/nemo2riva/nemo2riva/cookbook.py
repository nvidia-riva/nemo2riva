import gc
import os
import sys
import tempfile

import onnx
import onnx_graphsurgeon as gs
import torch
from eff.callbacks import BinaryContentCallback
from eff.core import Archive, ArtifactRegistry, File
from nemo.core import Exportable, ModelPT
from nemo.utils import logging
from .artifacts import create_artifact

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


class CudaOOMInExportOfASRWithMaxDim(Exception):
    def __init__(self, *args, max_dim=None):
        super().__init__(*args)
        self.max_dim = max_dim


def save_archive(model, save_path, cfg, artifacts, metadata):

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
        tmp_export_file = os.path.join(tmpdir, "export.onnx")
        if cfg.export_format in ["ONNX", "TS"]:
            # Export the model, get the descriptions.
            if not isinstance(model, Exportable):
                logging.error("Your NeMo model class ({}) is not Exportable.".format(metadata['obj_cls']))
                sys.exit(1)

            in_args = {}
            error_msg = (
                "ERROR: Export failed. Please make sure your NeMo model class ({}) has working export() and that "
                "you have the latest NeMo package installed with [all] dependencies.".format(metadata['obj_cls'])
            )
            try:
                autocast = nullcontext
                need_autocast = cfg.autocast
                if cfg.args.autocast is not None:
                    need_autocast = cfg.args.autocast
                if need_autocast:
                    autocast = torch.cuda.amp.autocast

                if cfg.args.max_batch is not None:
                    in_args["max_batch"] = cfg.args.max_batch

                # Set max_dim if specified in cfg
                if cfg.max_dim is not None:
                    in_args["max_dim"] = cfg.max_dim
                # Overide max_dim if specified in args
                if cfg.args.max_dim is not None:
                    in_args["max_dim"] = cfg.args.max_dim

                with autocast(), torch.inference_mode():
                    logging.info(f"Exporting model with autocast={need_autocast}")
                    model = model.to(device=cfg.args.device)
                    model.eval()
                    input_example = model.input_module.input_example(**in_args)
                    _, descriptions = model.export(
                        tmp_export_file,
                        input_example=input_example,
                        check_trace=cfg.args.runtime_check,
                        onnx_opset_version=cfg.args.onnx_opset,
                        verbose=cfg.args.verbose,
                    )
                    del model
                    model_onnx = onnx.load_model(tmp_export_file)
                    graph = gs.import_onnx(model_onnx)
                    graph.fold_constants().cleanup()
                    model_onnx = gs.export_onnx(graph)
                    onnx.save_model(model_onnx, export_file)
            except RuntimeError as e:
                if (
                    model.__class__.__name__ == 'EncDecCTCModelBPE'
                    and model.encoder.__class__.__name__ == 'ConformerEncoder'
                    and "max_dim" in in_args
                    and "CUDA out of memory" in str(e)
                ):
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
