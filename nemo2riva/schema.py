# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from eff.core import Archive
from eff.validator import schema_validate_archive
from nemo.package_info import __version__ as nemo_version
from nemo.utils import logging
from omegaconf import OmegaConf
from packaging.specifiers import SpecifierSet
from packaging.version import Version

schema_dict = None

supported_formats = ["ONNX", "CKPT", "TS", "NEMO", "PYTORCH", "STATE"]


@dataclass
class ExportConfig:
    """Default config model the model export to ONNX."""

    # Export format.
    export_subnet: str = ""
    export_format: str = "ONNX"
    export_file: str = "model_graph.onnx"
    encryption: Optional[str] = None
    autocast: bool = False
    max_dim: int = None
    export_args: Optional[Dict[str,str]] = None
    
@dataclass
class ImportConfig:
    """Default config model for the model that exports to ONNX."""

    exports = [None]
    # Encryption option.
    should_encrypt: bool = False
    validation_schema: Optional[str] = None


def get_export_config(export_obj, args):
    conf = ExportConfig()
    need_autocast = False
    if export_obj:
        conf.export_file = list(export_obj)[0]
        attribs = export_obj[conf.export_file]
        conf.export_subnet = attribs.get('export_subnet', None)
        conf.is_onnx=attribs.get('onnx', False)

        if not conf.is_onnx:
            conf.states_only = attribs.get('states_only', False)
            conf.is_torch = attribs.get('torch', False)

        if conf.export_file.endswith('.onnx'):
            conf.export_format = "ONNX"
        elif conf.export_file.endswith('.ts'):
            conf.export_format = "TS"
        elif conf.export_file.endswith('.nemo'):
            conf.export_format = "NEMO"
        elif conf.is_torch:
            if conf.states_only:
                conf.export_format = "STATE"
            else:
                conf.export_format = "PYTORCH"
        else:
            conf.export_format = "CKPT"
        conf.autocast = attribs.get('autocast', False)
        need_autocast = conf.autocast

        conf.max_dim = attribs.get('max_dim', None)

        conf.encryption = attribs.get('encryption', None)
        if conf.encryption and args.key is None:
            raise Exception(f"{conf.export_file} requires encryption and no key was given")

    if args.export_subnet:
        if conf.export_subnet:
            raise Exception("Can't combine schema's export_subnet and export-subnet argument!")
        conf.export_subnet = args.export_subnet

    if args.autocast is not None:
        need_autocast = args.autocast
        if need_autocast:
            autocast = torch.cuda.amp.autocast
    conf.autocast = need_autocast

    if args.max_dim is not None:
        conf.max_dim = args.max_dim

    # Optional export format override
    if args.format is not None:
        conf.export_format = args.format.upper()
        conf.export_file = os.path.splitext(conf.export_file)[0] + "." + conf.export_format.lower()

    if conf.export_format not in supported_formats:
        raise Exception(
            "Format `{}` is invalid. Please pick one of the ({})".format(conf.export_format, supported_formats)
        )

    # TODO: read from schema?
    conf.export_args = {}
    if args.export_config:
        for key_value in args.export_config:
            lst = key_value.split("=")
            if len(lst) != 2:
                raise Exception("Use correct format for --export_config: k=v")
            k, v = lst
            conf.export_args[k] = v

    if args.cache_support:
        conf.export_args.update({"cache_support": "True"})

    return conf


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


def get_subnet(model, subnet):
    submodel = None
    if subnet and subnet != 'self':
        try:
            # FIXME: remove special case once RNNT Nemo model provides this property
            if subnet == 'decoder_joint':
                from nemo.collections.asr.modules.rnnt import RNNTDecoderJoint

                submodel = RNNTDecoderJoint(model.decoder, model.joint)
            else:
                submodel = getattr(model, subnet, None)
        except Exception:
            pass
        if submodel is None:
            raise Exception("Failed to find subnetwork named: {} in {}.".format(subnet, model.__class__))
    else:
        submodel = model
    return submodel


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
        logging.info(f"Loaded schema file {f} for {key}")


def get_exports(schema_path):
    # Load the schema.
    schema = OmegaConf.load(schema_path)
    file_schemas = schema["file_properties"] if "file_properties" in schema else schema["artifact_properties"]
    exports = []
    for schema_section in file_schemas:
        try:
            for k in schema_section.keys():
                if os.path.splitext(k)[1].lower() in [".onnx", ".ckpt", ".pt", ".ts", '.nemo']:
                    exports.append(schema_section)
                    break
        except Exception:
            pass
    if len(exports) == 0:
        exports = [None]

    return exports


def get_import_config(model, args):

    # Explicit schema name passed in args
    schema = args.schema

    if schema_dict is None:
        load_schemas()

    # create config object with default values (ONNX)
    conf = ImportConfig()
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
        exports = [None]
    else:
        exports = get_exports(schema)

    conf.exports = [get_export_config(export_obj, args) for export_obj in exports]

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
