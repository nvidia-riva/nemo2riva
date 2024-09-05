import argparse
import gc
import json
import os
import tarfile
import tempfile
from contextlib import nullcontext

import click
import nemo.collections.asr.models as nemo_asr
import torch
import yaml
from eff.callbacks import BinaryContentCallback
from eff.core import Archive, ArtifactRegistry, File
from nemo2riva.artifacts import create_artifact
from nemo.core import Exportable, ModelPT
from nemo.utils import logging
from omegaconf import OmegaConf

TORCH_DTYPES = {
    'float32': torch.float32,
    'float64': torch.float64,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


class UnsupportedModel(Exception):
    def __init__(self, *args, model_name):
        super().__init__(*args)
        self.model_name = model_name


class CudaOOMInExportOfASRWithMaxDim(Exception):
    def __init__(self, *args, max_dim=None):
        super().__init__(*args)
        self.max_dim = max_dim


class CanaryModel:
    def __init__(self, model_name: str = 'nvidia/canary-1b', model: str = None, dtype='float32'):
        self.dtype = TORCH_DTYPES[dtype]
        self.artifacts = ArtifactRegistry(passphrase="tlt_encode")

        if model is not None:
            if self.is_supported(model_name):
                self.model_name = model_name
                try:
                    with torch.inference_mode():
                        # Restore instance from .nemo file using generic model restore_from
                        self.model = nemo_asr.EncDecMultiTaskModel.restore_from(model).to('cpu')

                except Exception as e:
                    logging.error(f"Failed to restore model from NeMo file : {model}. ")
                    raise e
        else:
            if model_name == 'nvidia/canary-1b':
                self.model = nemo_asr.EncDecMultiTaskModel.from_pretrained(model_name).to('cpu')
            else:
                raise UnsupportedModel(model_name=model_name)
        self.model.freeze()
        # self.export(rmir_file, export_encoder=export_encoder, export_decoder=export_decoder,export_vocab=export_vocab)

    def __str__(self):
        return "trtllm.canary"

    @staticmethod
    def is_supported(model_type: str):
        supported_models = ['nvidia/canary-1b', 'nemo.canary', 'nemo/canary', 'canary']

        if model_type in supported_models:
            return True

        return False

    def export(self, riva_file, export_encoder, export_decoder, export_vocab, export_config):
        with tempfile.TemporaryDirectory() as tempdir:
            # tempdir = '/tmp/temp2'
            if export_encoder:
                format_meta = {
                    "onnx": True,
                    "onnx_archive_format": 1,
                    "runtime": "ONNX",
                }
                encoder_path, encoder_filename = self.export_encoder(tempdir)
                create_artifact(
                    self.artifacts,
                    encoder_filename,
                    do_encrypt=False,
                    filepath=encoder_path,
                    description="Exported Encoder model",
                    content_callback=BinaryContentCallback,
                    **format_meta,
                )
            if export_decoder:
                format_meta = {
                    "torchscript": True,
                    "runtime": "TorchScript",
                }
                decoder_path, decoder_filename = self.export_decoder(tempdir)
                create_artifact(
                    self.artifacts,
                    decoder_filename,
                    do_encrypt=False,
                    filepath=decoder_path,
                    description="Exported Decoder model",
                    content_callback=BinaryContentCallback,
                    **format_meta,
                )
            if export_vocab:
                format_meta = {"conf_path": "vocab.json"}
                vocab_path, vocab_filename = self.export_vocab(tempdir)
                create_artifact(
                    self.artifacts,
                    vocab_filename,
                    do_encrypt=False,
                    filepath=vocab_path,
                    description="Exported Decoder model",
                    content_callback=BinaryContentCallback,
                    **format_meta,
                )

            if export_config:
                format_meta = {"conf_path": "model_config.yaml"}
                config_path, config_filename = self.export_config(tempdir)
                create_artifact(
                    self.artifacts,
                    config_filename,
                    do_encrypt=False,
                    filepath=config_path,
                    description="Exported Decoder model",
                    content_callback=BinaryContentCallback,
                    **format_meta,
                )

            metadata = {
                "description": "Exported NeMo Model",
                "format_version": 3,
                "has_pytorch_checkpoint": False,
                # use 'normalized' class name
                "obj_cls": 'trtllm.canary',
                "min_nemo_version": "1.23",
            }

            Archive.save_registry(save_path=riva_file, registry_name="artifacts", registry=self.artifacts, **metadata)

    def export_config(self, path):
        keys_required = [
            'beam_search',
            'encoder',
            'head',
            'model_defaults',
            'prompt_format',
            'sample_rate',
            'target',
            'preprocessor',
        ]

        config_file = "model_config.yaml"
        model_config = OmegaConf.to_container(self.model.cfg)

        if 'beam_search' not in model_config and 'decoding' in model_config:
            model_config['beam_search'] = model_config['decoding'].get('beam', {'beam_size': 1, 'len_pen': 0.0, 'max_generation_delta': 50}
)


        config = dict({k: model_config[k] for k in keys_required})
        config['decoder'] = {
            'transf_decoder': model_config['transf_decoder'],
            'transf_encoder': model_config['transf_encoder'],
            'vocabulary': self.model.tokenizer.vocab,
            'num_classes': model_config['head']['num_classes'],
            'feat_in': model_config['model_defaults']['asr_enc_hidden'],
            'n_layers': model_config['transf_decoder']['config_dict']['num_layers'],
        }
        config['target'] = 'trtllm.canary'

        config_path = os.path.join(path, config_file)
        with open(config_path, 'w') as ofp:
            yaml.dump(config, ofp, allow_unicode=True)

        return config_path, config_file

    def export_encoder(self, path):
        encoder_path = f"{path}/encoder"
        try:
            autocast = torch.cuda.amp.autocast(enabled=True, cache_enabled=False, dtype=self.dtype)
            with autocast, torch.no_grad(), torch.inference_mode():
                logging.info(f"Exporting model {self.model.__class__.__name__}")
                os.makedirs(encoder_path, exist_ok=True)
                encoder_filename = 'encoder.onnx'
                export_file = os.path.join(encoder_path, "encoder.onnx")

                self.model.encoder.export(export_file, onnx_opset_version=17)
                o_list = os.listdir(encoder_path)
                save_as_external_data = len(o_list) > 1
                if save_as_external_data:
                    o_list = os.listdir(encoder_path)
                    export_file = encoder_path + '.tar'
                    encoder_filename = 'encoder.tar'

                    logging.info(
                        f"Large (>2GB) ONNX is being exported with external weights, as {export_file} TAR archive!"
                    )
                    with tarfile.open(export_file, "w") as tar:
                        for f in o_list:
                            fpath = os.path.join(encoder_path, f)
                            tar.add(fpath, f)

        except Exception as e:
            raise e

        return export_file, encoder_filename

    def export_decoder(self, path):
        try:
            decoder_filename = 'decoder.pt'
            decoder_path = os.path.join(path, decoder_filename)
            self.model.transf_decoder.freeze()
            decoder_params = self.model.transf_decoder.state_dict()
            decoder_params.update(self.model.log_softmax.state_dict())

            torch.save(decoder_params, decoder_path)

        except Exception as e:
            raise e

        return decoder_path, decoder_filename

    def export_vocab(self, path):
        vocab_file = "vocab.json"
        vocab_path = os.path.join(path, vocab_file)
        tokenizer_vocab={'tokens':{},
                         'offsets':self.model.tokenizer.token_id_offset
                         }
        for lang in self.model.tokenizer.langs:
            tokenizer_vocab['tokens'][lang] = {}
        tokenizer_vocab['size']=self.model.tokenizer.vocab_size

        try:
            tokenizer_vocab['bos_id']=self.model.tokenizer.bos_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing bos_id. Could affect accuracy")

        try:
            tokenizer_vocab['eos_id']=self.model.tokenizer.eos_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing eos_id. Could affect accuracy")
        try:
            tokenizer_vocab['nospeech_id']=self.model.tokenizer.nospeech_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing nospeech_id. Could affect accuracy")
        try:
            tokenizer_vocab['pad_id'] = self.model.tokenizer.pad_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing pad_id. Could affect accuracy")


        for t_id in range(0, self.model.tokenizer.vocab_size):
            lang = self.model.tokenizer.ids_to_lang([t_id])
            tokenizer_vocab['tokens'][lang][t_id] = self.model.tokenizer.ids_to_tokens([t_id])[0]

        with open(vocab_path, 'w') as ofp:
            json.dump(tokenizer_vocab, ofp)

        return vocab_path, vocab_file


@click.command()
@click.option('--model', type=str, default='nvidia/canary-1b')
@click.option('--model_file', type=str, default=None)
@click.option('--dtype', type=str, default='float32')
@click.option('--export_encoder', is_flag=True, default=True)
@click.option('--export_decoder', is_flag=True, default=True)
@click.option('--export_vocab', is_flag=True, default=True)
@click.option('--export_config', is_flag=True, default=True)
@click.argument('riva_file', type=click.Path(dir_okay=False))
def export(riva_file, model, model_file, dtype, export_encoder, export_decoder, export_vocab, export_config):
    CanaryModel(model, model_file, dtype=dtype).export(
        riva_file, export_encoder, export_decoder, export_vocab, export_config
    )


if __name__ == "__main__":
    export()
