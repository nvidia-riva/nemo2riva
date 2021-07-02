# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

from eff.cookbooks import NeMoCookbook, ONNXCookbook
from eff.core import Archive
from omegaconf import OmegaConf
from packaging import version

from nemo.package_info import __version__ as nemo_version

schema_dict = None

supported_formats = ["ONNX", "CKPT", "TS"]


@dataclass
class ExportConfig:
    """Default config model the model export to ONNX."""

    # Export format.
    export_format: str = "ONNX"

    export_file: str = "model_graph.onnx"

    # Encryption option.
    should_encrypt: bool = False

    validation_schema: Optional[str] = None


def get_schema_key(model):
    key = model.cfg.target
    # normalize the key: remove extra qualifiers
    keylist = key.split('.')
    try:
        model_index = keylist.index('models')
        key = '.'.join(keylist[: model_index + 1]) + '.' + keylist[-1]
    except ValueError:
        pass
    if (
        key.startswith('nemo.collections.nlp')
        and hasattr(model.cfg, "language_model")
        and 'megatron' in model.cfg.language_model.pretrained_model_name
    ):
        key = key + "-megatron"
    return key


def load_schemas():
    spec_root = os.path.dirname(os.path.abspath(__file__))
    # Get schema path.
    direc = os.path.join(spec_root, "validation_schemas")
    ext = '.yaml'

    global schema_dict
    schema_dict = {}  # Create an empty dict

    # Select only .yaml files
    yaml_files = [os.path.join(direc, i) for i in os.listdir(direc) if os.path.splitext(i)[1] == '.yaml']

    # Iterate over your txt files
    for f in yaml_files:
        conf = OmegaConf.load(f)
        key = ''
        for meta in conf.metadata:
            if 'obj_cls' in meta.keys():
                key = meta['obj_cls']
        if 'megatron' in f:
            key = key + "-megatron"
        schema_dict[key] = f
        print(f"Loaded schema file {f} for {key}")


def get_export_format(schema_path):
    # Load the schema.
    schema = OmegaConf.load(schema_path)
    obj = {'model_graph.onnx': {'onnx': True, 'encrypted': False}}

    for schema_section in schema["file_properties"]:
        try:
            if (
                "model_graph.onnx" in schema_section
                or "model_weights.ckpt" in schema_section
                or "model_graph.ts" in schema_section
            ):
                return schema_section
        except Exception:
            pass

    return obj


def get_export_config(model, args):

    # Explicit schema name passed in args
    schema = args.schema

    if schema_dict is None:
        load_schemas()

    # create config object with default values (ONNX)
    conf = ExportConfig()

    key = get_schema_key(model)

    #
    # Current 'state of that art' of export formats Riva expects is:
    # ONNX for all models except mon-Megatron NLP models
    #
    if key.startswith('nemo.collections.nlp') and not 'megatron' in key:
        conf.export_format = 'CKPT'
        conf.export_file = 'model_graph.ckpt'

    #
    # Now check if there is a schema defined for target model class
    #
    if schema is None and key in schema_dict:
        schema = schema_dict[key]
        logging.info("Found validation schema for {} at {}".format(key, schema))

    if schema is not None:
        export_obj = get_export_format(schema)
        conf.export_file = list(export_obj)[0]
        if conf.export_file.endswith('.onnx'):
            conf.export_format = "ONNX"
        elif conf.export_file.endswith('.ts'):
            conf.export_format = "TS"
        else:
            conf.export_format = "CKPT"
        # conf.should_encrypt = export_obj[conf.export_file]['encrypted']

    # Optional export format override
    if args.format is not None:
        conf.export_format = args.format.upper()
        conf.export_file = os.path.splitext(conf.export_file)[0] + "." + conf.export_format.lower()

    if conf.export_format not in supported_formats:
        raise Exception(
            "Format `{}` is invalid. Please pick one of the ({})".format(conf.export_format, supported_formats)
        )

    conf.validation_schema = schema
    conf.args = args
    check_nemo_version(conf)

    return conf


def check_nemo_version(conf):
    """
       Check if installed NeMo version is sufficient to restore/export model.
    """
    if conf.validation_schema is not None:
        schema = OmegaConf.load(conf.validation_schema)
        key = ''
        min_nemo_version = None

        for meta in schema.metadata:
            if 'min_nemo_version' in meta.keys():
                min_nemo_version = meta['min_nemo_version']

        if min_nemo_version is not None:
            if version.parse(str(min_nemo_version)) > version.parse(nemo_version):
                msg = "This model requires NeMo version >= {}, you have {} installed. Please install NeMo >= {}.".format(
                    min_nemo_version, nemo_version, min_nemo_version
                )
                if conf.args.validate:
                    logging.error(msg)
                    sys.exit(1)
                else:
                    logging.warning(msg)
            else:
                logging.info("Checking installed NeMo version ... {} OK".format(nemo_version))


def validate_archive(save_path, schema):
    """
    Validate EFF archive at save_path using the schema.
    """
    if schema is None:
        logging.error("--validate option used, but no --schema and no default schema found!")

    logging.warning("EFF file content validation is disabled in this release!")
    return

    if Archive.schema_validate_archive(archive_path=save_path, schema_path=schema):
        logging.info(f"Exported model at {save_path} is compliant with Riva, using schema at {schema}")
    else:
        logging.error(f"Exported model at {save_path} failed Riva compliance, using schema at {schema} !")
