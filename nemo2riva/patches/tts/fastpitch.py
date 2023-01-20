# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import nemo
import torch
import yaml
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

from nemo2riva.patches.tts.general import create_batch

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
