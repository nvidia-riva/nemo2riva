import logging
import sys
from typing import Optional

import nemo
import torch
import wrapt
import yaml
from nemo.collections.tts.helpers.helpers import regulate_len
from nemo.collections.tts.torch.tts_tokenizers import IPATokenizer
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


@torch.jit.script
def create_batch(
    text: torch.Tensor,
    pitch: torch.Tensor,
    pace: torch.Tensor,
    batch_lengths: torch.Tensor,
    padding_idx: int = -1,
    volume: Optional[torch.Tensor] = None,
):
    batch_lengths = batch_lengths.to(torch.int64)
    max_len = torch.max(batch_lengths[1:] - batch_lengths[:-1])

    index = 1
    texts = torch.zeros(batch_lengths.shape[0] - 1, max_len, dtype=torch.int64, device=text.device) + padding_idx
    pitches = torch.zeros(batch_lengths.shape[0] - 1, max_len, dtype=torch.float32, device=text.device)
    paces = torch.zeros(batch_lengths.shape[0] - 1, max_len, dtype=torch.float32, device=text.device) + 1.0
    volumes = torch.zeros(batch_lengths.shape[0] - 1, max_len, dtype=torch.float32, device=text.device) + 1.0

    while index < batch_lengths.shape[0]:
        seq_start = batch_lengths[index - 1]
        seq_end = batch_lengths[index]
        cur_seq_len = seq_end - seq_start

        texts[index - 1, :cur_seq_len] = text[seq_start:seq_end]
        pitches[index - 1, :cur_seq_len] = pitch[seq_start:seq_end]
        paces[index - 1, :cur_seq_len] = pace[seq_start:seq_end]
        if volume is not None:
            volumes[index - 1, :cur_seq_len] = volume[seq_start:seq_end]

        index += 1

    return texts, pitches, paces, volumes


def generate_vocab_mapping_arpabet(labels):
    mapping = []
    for idx, token in enumerate(labels):
        # Patch to remove emphasis from 22.08 TTS model
        # if token == "[" or token == "]":
        #     continue
        if not str.islower(token) and str.isalnum(token):
            # token is ARPABET token, need to be prepended with @
            token = '@' + token
        mapping.append("{} {}".format(idx, token))
        if str.islower(token) and str.isalnum(token):
            # normal lowercase token, we want to create uppercase variant too
            # since nemo preprocessing includes a .tolower
            mapping.append("{} {}".format(idx, token.upper()))
    return mapping


def generate_vocab_mapping_ipa(labels):
    # Only support English IPA dict
    VALID_NON_ALNUM_IPA_TOKENS = ['ˈ', 'ˌ']
    mapping = []
    for idx, token in enumerate(labels):
        if token in VALID_NON_ALNUM_IPA_TOKENS or (str.isalnum(token) and str.islower(token)):
            # This is a phone
            token = '@' + token
        mapping.append("{} {}".format(idx, token))
    return mapping


def generate_vocab_mapping(model, artifacts, **kwargs):
    # TODO Hack to add labels from FastPitch to .riva since that file is not inside the .nemo
    # Task tracked at https://jirasw.nvidia.com/browse/JARS-1169
    if model.__class__.__name__ == 'FastPitchModel' and hasattr(model, 'vocab'):
        logging.info("Adding mapping.txt for FastPitchModel instance to output file")
        ipa_support = False
        if hasattr(model.vocab, "labels"):
            labels = model.vocab.labels
        else:
            labels = model.vocab.tokens
            if isinstance(model.vocab, IPATokenizer):
                ipa_support = True

        if ipa_support:
            mapping = generate_vocab_mapping_ipa(labels)
        else:
            mapping = generate_vocab_mapping_arpabet(labels)

        mapping_txt = "\n".join(mapping).encode('utf-8')

        content = {
            "description": "mapping file for FastPitch",
            "path_type": "TAR_PATH",
            "nemo_artifact": True,
            "content": mapping_txt,
        }
        artifacts["mapping.txt"] = content


def fastpitch_model_versioning(model, artifacts, **kwargs):
    # Riva supports some additional features over NeMo fastpitch models depending on the version
    # Namely, we need to patch in volume support and ragged batched support for lower NeMo versions
    try:
        nemo_version = Version(nemo.__version__)
    except NameError:
        # If can't find the nemo version, return without patching
        return None
    if model.__class__.__name__ == 'FastPitchModel':
        if nemo_version < Version('1.11.0'):
            # If nemo_version is less than 1.10, we need to manually add the volume updates
            # and the ragged batch updates

            # Patch model's _prepare_for_export()
            def _prepare_for_export(self, **kwargs):
                super(model.__class__, model)._prepare_for_export(**kwargs)

                # Define input_types and output_types as required by export()
                self._input_types = {
                    "text": NeuralType(('T'), TokenIndex()),
                    "pitch": NeuralType(('T'), RegressionValuesType()),
                    "pace": NeuralType(('T')),
                    "volume": NeuralType(('T'), optional=True),
                    "batch_lengths": NeuralType(('B'), optional=True),
                    "speaker": NeuralType(('B'), Index(), optional=True),
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
            def forward_for_export(self, text, pitch, pace, volume, batch_lengths, speaker=None):
                text, pitch, pace, volume = create_batch(
                    text, pitch, pace, batch_lengths, padding_idx=self.fastpitch.encoder.padding_idx, volume=volume
                )
                try:
                    return self.fastpitch.infer(text=text, pitch=pitch, pace=pace, volume=volume, speaker=speaker)
                except TypeError as e:
                    if 'volume' in str(e):
                        # NeMo version <= 1.9.0 when we don't return volume
                        base_return = self.fastpitch.infer(text=text, pitch=pitch, pace=pace, speaker=speaker)
                        durs_predicted = base_return[2]
                        volume_extended, _ = regulate_len(durs_predicted, volume.unsqueeze(-1), pace)
                        volume_extended = volume_extended.squeeze(-1).float()
                        return (*base_return, volume_extended)

            model.__class__.forward_for_export = forward_for_export

            # Patch module's input_example()
            def input_example(self, max_batch=1, max_dim=44):
                par = next(self.fastpitch.parameters())
                sz = (max_batch * max_dim,)
                inp = torch.randint(
                    0, self.fastpitch.encoder.word_emb.num_embeddings, sz, device=par.device, dtype=torch.int64
                )
                pitch = torch.randn(sz, device=par.device, dtype=torch.float32) * 0.5
                pace = torch.clamp(torch.randn(sz, device=par.device, dtype=torch.float32) * 0.1 + 1, min=0.01)

                inputs = {'text': inp, 'pitch': pitch, 'pace': pace}

                volume = torch.clamp(torch.randn(sz, device=par.device, dtype=torch.float32) * 0.1 + 1, min=0.01)
                inputs['volume'] = volume
                batch_lengths = torch.zeros((max_batch + 1), device=par.device, dtype=torch.int32)
                left_over_size = sz[0]
                batch_lengths[0] = 0
                for i in range(1, max_batch):
                    length = torch.randint(1, left_over_size - (max_batch - i), (1,), device=par.device)
                    batch_lengths[i] = length + batch_lengths[i - 1]
                    left_over_size -= length.detach().cpu().numpy()[0]
                batch_lengths[-1] = left_over_size + batch_lengths[-2]

                sum = 0
                index = 1
                while index < len(batch_lengths):
                    sum += batch_lengths[index] - batch_lengths[index - 1]
                    index += 1
                assert sum == sz[0], f"sum: {sum}, sz: {sz[0]}, lengths:{batch_lengths}"
                inputs['batch_lengths'] = batch_lengths

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
        else:
            # NeMo version >= 1.11.0; can just set the relevant flags
            model.export_config["enable_volume"] = True
            model.export_config["enable_ragged_batches"] = True

        # Patch the model config yaml to add the volume and ragged batch flags
        for art in artifacts:
            if art == 'model_config.yaml':
                model_config = yaml.safe_load(artifacts['model_config.yaml']['content'])
                model_config["export_config"] = {'enable_volume': True, 'enable_ragged_batches': True}
                artifacts['model_config.yaml']['content'] = yaml.dump(model_config).encode()
