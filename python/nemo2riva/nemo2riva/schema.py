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

from eff.core import Archive
from eff.validator import schema_validate_archive
from nemo.package_info import __version__ as nemo_version
from omegaconf import OmegaConf
from packaging.specifiers import SpecifierSet
from packaging.version import Version

schema_dict = None

supported_formats = ["ONNX", "CKPT", "TS"]


@dataclass
class ExportConfig:
    """Default config model the model export to ONNX."""

    # Export format.
    export_format: str = "ONNX"
    autocast: bool = False
    export_file: str = "model_graph.onnx"

    # Encryption option.
    should_encrypt: bool = False
    encryption: Optional[str] = None
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
        schema_dict[key] = f
        print(f"Loaded schema file {f} for {key}")


def get_export_format(schema_path):
    # Load the schema.
    schema = OmegaConf.load(schema_path)
    obj = {'model_graph.onnx': {'onnx': True, 'autocast': False}}
    file_schemas = schema["file_properties"] if "file_properties" in schema else schema["artifact_properties"]
    for schema_section in file_schemas:
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
    # Now check if there is a schema defined for target model class
    #
    if schema is None and key in schema_dict:
        schema = schema_dict[key]
        logging.info("Found validation schema for {} at {}".format(key, schema))

    if schema is None:
        logging.warning(
            "Validation schema not found for {}.\n".format(key)
            + "That means Riva does not yet support a pipeline for this network and likely will not work with it."
        )
    else:
        export_obj = get_export_format(schema)
        conf.export_file = list(export_obj)[0]
        if conf.export_file.endswith('.onnx'):
            conf.export_format = "ONNX"
        elif conf.export_file.endswith('.ts'):
            conf.export_format = "TS"
        else:
            conf.export_format = "CKPT"
        conf.autocast = export_obj[conf.export_file].get('autocast', False)
        conf.encryption = export_obj[conf.export_file].get('encryption', None)

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
    # save 'normalized' class name
    conf.cls = key

    conf_check_nemo_version(conf)

    return conf


def check_nemo_version(spec_str, error=True, prereleases=True):
    """
       Check if installed NeMo version conforms to spec_str
    """
    spec = SpecifierSet(spec_str, prereleases=prereleases)
    if nemo_version in spec:
        logging.info("Checking installed NeMo version ... {} OK ({})".format(nemo_version, spec_str))
    else:
        msg = "This model requires nemo_toolkit package version {}, you have {} installed.\nPlease install nemo_toolkit {}.".format(
            spec_str, nemo_version, spec_str
        )
        if error:
            logging.error(msg)
            sys.exit(1)
        else:
            logging.warning(msg)


def conf_check_nemo_version(conf):
    """
       Check if installed NeMo version conforms to model config
    """
    spec_str = None
    if conf.validation_schema is not None:
        schema = OmegaConf.load(conf.validation_schema)
        for meta in schema.metadata:
            if 'min_nemo_version' in meta.keys():
                min_version = meta['min_nemo_version']
                spec_str = f">={min_version}"
                break

    if spec_str is not None:
        check_nemo_version(spec_str, conf.args.validate)


def validate_archive(save_path, schema):
    """
    Validate EFF archive at save_path using the schema.
    """
    if schema is None:
        logging.error("--validate option used, but no --schema and no default schema found!")

    if schema_validate_archive(archive_path=save_path, schema_path=schema):
        logging.info(f"Exported model at {save_path} is compliant with Riva, using schema at {schema}")
    else:
        logging.error(f"Exported model at {save_path} failed Riva compliance, using schema at {schema} !")
