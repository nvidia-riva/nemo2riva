import os

import torch

from eff.core import Archive, Cookbook, Expression, Origins, Runtimes
from nemo.core import ModelPT


class Nemo2RivaCookbook(Cookbook):

    # Class attribute: encryption key - a class property shared between all objects.
    _encryption_key = None

    # Class attribute: additional medatata that will be added to any instantiated object of that class.
    _class_metadata = {}

    # Class attribute: additional files that will be added to any instantiated object of that class.
    _class_file_content = {}

    def save(self, obj, save_path, cfg):
        runtime = Runtimes.PyTorch

        # Properties:
        common_meta = {
            "description": "This format stores the whole model in a single {} graph.".format(cfg.export_format),
            "obj_cls": Archive.generate_obj_cls(obj),
            "origin": Origins.NeMo,
            # Indicate that it has config file.
            "has_nemo_config": True,
        }

        if cfg.export_format == "TS":
            format_meta = {
                "runtime": Runtimes.PyTorch,
                "nemo_archive_version": 2,
            }
        elif cfg.export_format == "ONNX":
            format_meta = {
                "runtime": Runtimes.ONNX,
                "onnx_archive_format": 1,
            }
        elif cfg.export_format == "CKPT":
            format_meta = {
                "runtime": Runtimes.PyTorch,
                "has_pytorch_checkpoint": True,
            }

        # Create EFF archive.
        with Archive.create(
            save_path=save_path, encryption_key=self.get_encryption_key(), **common_meta, **format_meta,
        ) as effa:

            # Add additional metadata stored by the NeMoCookbook class.
            effa.add_metadata(force=True, **self.class_metadata)

            if cfg.export_format in ["ONNX", "TS"]:
                # Add exported file to the archive.
                model_graph = effa.create_file_handle(
                    name=cfg.export_file,
                    description="Exported model graph",
                    encrypted=(self.get_encryption_key() is not None),
                    onnx=True,
                )

                # Export the model, get the descriptions.
                _, descriptions = obj.export(model_graph, check_trace=cfg.args.runtime_check)

                # Overwrite the file description.
                effa.add_file_properties(name=cfg.export_file, force=True, description=descriptions[0])

            elif cfg.export_format == "CKPT":
                # Add model weights to archive - encrypt when the encryption key is provided.
                model_weights = effa.create_file_handle(
                    name=cfg.export_file,
                    description="File containing model weights",
                    encrypted=(self.get_encryption_key() is not None),
                )
                # Save model state using torch save.
                torch.save(obj.state_dict(), model_weights)

            # Add artifacts
            for filename, (content, props) in self.class_file_content.items():
                # Create handle and pass properties - potentially overwrite the old files with new ones.
                file_handle = effa.create_file_handle(name=filename, force=True, **props)
                # Write content depending on its type (bytes vs strings).
                write_mode = "wb" if type(content) == bytes else "w"
                with open(file_handle, write_mode) as f:
                    f.write(content)
