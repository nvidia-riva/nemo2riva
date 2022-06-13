import logging
import sys

import nemo
import torch
import wrapt
from nemo.collections.tts.helpers.helpers import regulate_len
from nemo.core.neural_types.elements import (
    Index,
    MelSpectrogramType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from packaging.version import Version

# Note: No need to mock as we don't support nemo2riva in our servicemaker container. Nemo2riva
#       should be done inside the NeMo training container.
# # fmt: off
# # Need to mock pillow as we uninstall pillow in our docker
# # All import pillow calls are from `import matplotlib` and since nemo2riva doesn't use matplotlib, we should be safe to just mock it
# from mock import MagicMock
# sys.modules["PIL"] = MagicMock()
# sys.modules["PIL.PngImagePlugin"] = MagicMock()
# # fmt: on


def generate_vocab_mapping(model, artifacts, **kwargs):
    # TODO Hack to add labels from FastPitch to .riva since that file is not inside the .nemo
    # Task tracked at https://jirasw.nvidia.com/browse/JARS-1169
    if model.__class__.__name__ == 'FastPitchModel' and hasattr(model, 'vocab'):
        logging.info("Adding mapping.txt for FastPitchModel instance to output file")
        if hasattr(model.vocab, "labels"):
            labels = model.vocab.labels
        else:
            labels = model.vocab.tokens
        mapping = []
        for idx, token in enumerate(labels):
            if not str.islower(token) and str.isalnum(token):
                # token is ARPABET token, need to be prepended with @
                token = '@' + token
            mapping.append("{} {}".format(idx, token))
            if str.islower(token) and str.isalnum(token):
                # normal lowercase token, we want to create uppercase variant too
                # since nemo preprocessing includes a .tolower
                mapping.append("{} {}".format(idx, token.upper()))

        mapping_txt = "\n".join(mapping).encode('utf-8')

        content = {
            "description": "mapping file for FastPitch",
            "path_type": "TAR_PATH",
            "nemo_artifact": True,
            "content": mapping_txt,
        }
        artifacts["mapping.txt"] = content


def patch_volume(model, artifacts, **kwargs):
    # Add volume ability for models made in NeMo <= 1.9.0
    try:
        nemo_version = Version(nemo.__version__)
    except NameError:
        # If can't find the nemo version, return without patching
        return None
    if model.__class__.__name__ == 'FastPitchModel' and nemo_version < Version('1.10.0'):
        fp_input_example = model.input_example()[0]
        if 'volume' not in fp_input_example:
            # Patch model's _prepare_for_export()
            def _prepare_for_export(self, **kwargs):
                super(model.__class__, model)._prepare_for_export(**kwargs)

                # Define input_types and output_types as required by export()
                self._input_types = {
                    "text": NeuralType(('B', 'T_text'), TokenIndex()),
                    "pitch": NeuralType(('B', 'T_text'), RegressionValuesType()),
                    "pace": NeuralType(('B', 'T_text'), optional=True),
                    "volume": NeuralType(('B', 'T_text')),
                    "speaker": NeuralType(('B'), Index()),
                }
                self._output_types = {
                    "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
                    "num_frames": NeuralType(('B'), TokenDurationType()),
                    "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
                    "log_durs_predicted": NeuralType(('B', 'T_text'), TokenLogDurationType()),
                    "pitch_predicted": NeuralType(('B', 'T_text'), RegressionValuesType()),
                    "volume_aligned": NeuralType(('B', 'T_spec'), RegressionValuesType()),
                }

            model.__class__._prepare_for_export = _prepare_for_export
            # Patch module's infer()
            def forward_for_export(self, text, pitch, pace, volume, speaker=None):
                base_return = self.fastpitch.infer(text=text, pitch=pitch, pace=pace, speaker=speaker)
                durs_predicted = base_return[2]
                assert volume is not None
                volume_extended, _ = regulate_len(durs_predicted, volume.unsqueeze(-1), pace)
                volume_extended = volume_extended.squeeze(-1).float()
                return (*base_return, volume_extended)

            model.__class__.forward_for_export = forward_for_export
            # Patch module's input_example()
            def input_example(self, max_batch=1, max_dim=128):
                par = next(self.fastpitch.parameters())
                sz = (max_batch, max_dim)
                inp = torch.randint(
                    0, self.fastpitch.encoder.word_emb.num_embeddings, sz, device=par.device, dtype=torch.int64
                )
                pitch = torch.randn(sz, device=par.device, dtype=torch.float32) * 0.5
                pace = torch.clamp((torch.randn(sz, device=par.device, dtype=torch.float32) + 1) * 0.1, min=0.01)
                volume = torch.clamp((torch.randn(sz, device=par.device, dtype=torch.float32) + 1) * 0.1, min=0.01)

                inputs = {'text': inp, 'pitch': pitch, 'pace': pace, 'volume': volume}

                if self.fastpitch.speaker_emb is not None:
                    inputs['speaker'] = torch.randint(
                        0,
                        self.fastpitch.speaker_emb.num_embeddings,
                        (max_batch,),
                        device=par.device,
                        dtype=torch.int64,
                    )

                return (inputs,)

            model.__class__.input_example = input_example
