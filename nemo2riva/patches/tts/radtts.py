# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import random
import nemo
import torch
import yaml
from nemo.core.neural_types.neural_type import NeuralType
from packaging.version import Version

from nemo2riva.patches.tts.general import create_batch

def radtts_model_versioning(model, artifacts, **kwargs):
    # Riva supports some additional features over NeMo radtts models depending on the version
    # Namely, we need to patch in
    # - pitch support
    # - pace support
    # - volume support
    # - ragged batching support. Not supported with torch backend, tracked as DLIS-4332
    try:
        nemo_version = Version(nemo.__version__)
    except NameError:
        # If can't find the nemo version, return without patching
        return None
    if model.__class__.__name__ == 'RadTTSModel':
        from nemo.collections.tts.helpers.helpers import regulate_len
        from nemo.collections.tts.modules.radtts import RadTTSModule, adjust_f0, pad_dur, pad_energy_avg_and_f0
        from nemo.core.neural_types.elements import (
            Index,
            LengthsType,
            MelSpectrogramType,
            RegressionValuesType,
            TokenDurationType,
            TokenIndex,
        )
        if nemo_version < Version('1.17.0') and not hasattr(model, "export_config"):
            # If nemo_version is less than 1.17, we need to add all supports

            # Patch model's _prepare_for_export()
            def _prepare_for_export(self, **kwargs):
                super(model.__class__, model)._prepare_for_export(**kwargs)
                # Define input_types and output_types as required by export()
                self._input_types = {
                    "text": NeuralType(('B', 'T'), TokenIndex()),
                    "batch_lengths": NeuralType(('B')),
                    # "batch_lengths": NeuralType(('B'), LengthsType(), optional=True),
                    "speaker_id": NeuralType(('B'), Index()),
                    "speaker_id_text": NeuralType(('B'), Index()),
                    "speaker_id_attributes": NeuralType(('B'), Index()),
                    "pitch": NeuralType(('B', 'T'), RegressionValuesType()),
                    "pace": NeuralType(('B', 'T')),
                    "volume": NeuralType(('B', 'T'), optional=True),
                }
                self._output_types = {
                    "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
                    "num_frames": NeuralType(('B'), TokenDurationType()),
                    "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
                    "volume_aligned": NeuralType(('B', 'T_spec'), RegressionValuesType()),
                }

            model.__class__._prepare_for_export = _prepare_for_export

            # Patch module's infer()
            def forward_for_export(
                # self, text, batch_lengths, speaker_id, speaker_id_text, speaker_id_attributes, pitch, pace, volume
                self,
                text,
                lens,
                speaker_id,
                speaker_id_text,
                speaker_id_attributes,
                pitch,
                pace,
                volume,
            ):
                # text, pitch, pace, volume = create_batch(
                #     text, pitch, pace, batch_lengths, padding_idx=self.tokenizer.pad, volume=volume
                # )
                (mel, n_frames, dur, _, _) = self.model.infer(
                    speaker_id,
                    text,
                    speaker_id_text=speaker_id_text,
                    speaker_id_attributes=speaker_id_attributes,
                    sigma=0.7,
                    sigma_txt=0.7,
                    sigma_f0=1.0,
                    sigma_energy=1.0,
                    f0_mean=0.0,
                    f0_std=0.0,
                    in_lens=lens,
                    pitch_shift=pitch,
                    pace=pace,
                ).values()
                volume_extended = volume
                # Need to reshape as in infer patch
                durs_predicted = dur.float()
                truncated_length = torch.max(lens)
                try:
                    # Use NeMo 1.16's function signature if possible
                    volume_extended, _ = regulate_len(
                        durs_predicted,
                        volume[:, :truncated_length].unsqueeze(-1),
                        pace[:, :truncated_length],
                        replicate_to_nearest_multiple=True,
                        group_size=self.model.n_group_size,
                        in_lens=lens,
                    )
                except TypeError as e:
                    # Else, default to NeMo 1.15's function signature
                    volume_extended, _ = regulate_len(
                        durs_predicted,
                        volume[:, :truncated_length].unsqueeze(-1),
                        pace[:, :truncated_length]
                    )
                volume_extended = volume_extended.squeeze(-1).float()
                return mel.float(), n_frames, dur.float(), volume_extended

            model.__class__.forward_for_export = forward_for_export

            def infer(
                self,
                speaker_id,
                text,
                sigma=0.7,
                sigma_txt=0.7,
                sigma_f0=1.0,
                sigma_energy=1.0,
                speaker_id_text=None,
                speaker_id_attributes=None,
                pace=None,
                token_duration_max=100,
                in_lens=None,
                dur=None,
                f0=None,
                f0_mean=0.0,
                f0_std=0.0,
                energy_avg=None,
                voiced_mask=None,
                pitch_shift=None,
            ):

                batch_size = text.shape[0]
                if in_lens is None:
                    in_lens = text.new_ones((batch_size,), dtype=torch.int64) * text.shape[1]
                    txt_len_pad_removed = text.shape[1]
                else:
                    txt_len_pad_removed = torch.max(in_lens)
                    # borisf : this should not be needed as long as we have properly formed input batch
                    text = text[:, :txt_len_pad_removed]

                spk_vec = self.encode_speaker(speaker_id)

                if speaker_id_text is None:
                    speaker_id_text = speaker_id
                if speaker_id_attributes is None:
                    speaker_id_attributes = speaker_id
                spk_vec_text = self.encode_speaker(speaker_id_text)
                spk_vec_attributes = self.encode_speaker(speaker_id_attributes)
                txt_enc, _ = self.encode_text(text, in_lens)

                if dur is None:
                    # get token durations
                    dur = self.dur_pred_layer.infer(txt_enc, spk_vec_text, lens=in_lens)
                    dur = pad_dur(dur, txt_enc)
                    dur = dur[:, 0]
                    dur = dur.clamp(0, token_duration_max)

                if pace is None:
                    pace = txt_enc.new_ones((batch_size, txt_len_pad_removed))
                else:
                    pace = pace[:, :txt_len_pad_removed]

                try:
                    txt_enc_time_expanded, out_lens = regulate_len(
                        dur,
                        txt_enc.transpose(1, 2),
                        pace,
                        group_size=self.n_group_size,
                        dur_lens=in_lens,
                    )
                except TypeError as e:
                    txt_enc_time_expanded, out_lens = regulate_len(
                        dur, txt_enc.transpose(1, 2), pace
                    )
                n_groups = torch.div(out_lens, self.n_group_size, rounding_mode='floor')
                max_out_len = torch.max(out_lens)

                txt_enc_time_expanded.transpose_(1, 2)
                if voiced_mask is None:
                    if self.use_vpred_module:
                        # get logits
                        voiced_mask = self.v_pred_module.infer(txt_enc_time_expanded, spk_vec_attributes, lens=out_lens)
                        voiced_mask_bool = torch.sigmoid(voiced_mask[:, 0]) > 0.5
                        voiced_mask = voiced_mask_bool.to(dur.dtype)
                    else:
                        voiced_mask_bool = None
                else:
                    voiced_mask_bool = voiced_mask.bool()

                ap_txt_enc_time_expanded = txt_enc_time_expanded
                # voice mask augmentation only used for attribute prediction
                if self.ap_use_voiced_embeddings:
                    ap_txt_enc_time_expanded = self.apply_voice_mask_to_text(txt_enc_time_expanded, voiced_mask)

                f0_bias = 0
                # unvoiced bias forward pass
                if self.use_unvoiced_bias:
                    f0_bias = self.unvoiced_bias_module(txt_enc_time_expanded.permute(0, 2, 1))
                    f0_bias = -f0_bias[..., 0]

                if f0 is None:
                    f0 = self.infer_f0(ap_txt_enc_time_expanded, spk_vec_attributes, voiced_mask_bool, out_lens)[:, 0]

                f0 = adjust_f0(f0, f0_mean, f0_std, voiced_mask_bool)

                if energy_avg is None:
                    energy_avg = self.infer_energy(ap_txt_enc_time_expanded, spk_vec, out_lens)[:, 0]

                # replication pad, because ungrouping with different group sizes
                # may lead to mismatched lengths
                # FIXME: use replication pad
                (energy_avg, f0) = pad_energy_avg_and_f0(energy_avg, f0, max_out_len)

                if pitch_shift is not None:
                    try:
                        pitch_shift_spec_len, _ = regulate_len(
                            dur,
                            pitch_shift[:, :txt_len_pad_removed].unsqueeze(-1),
                            pace,
                            group_size=self.n_group_size,
                            dur_lens=in_lens,
                        )
                    except TypeError as e:
                        pitch_shift_spec_len, _ = regulate_len(
                            dur,
                            pitch_shift[:, :txt_len_pad_removed].unsqueeze(-1),
                            pace,
                        )
                    f0_bias = pitch_shift_spec_len.squeeze(-1) + f0_bias

                context_w_spkvec = self.preprocess_context(
                    txt_enc_time_expanded, spk_vec, out_lens, (f0 + f0_bias) * voiced_mask, energy_avg, assume_padded=True,
                )

                residual = txt_enc.new_zeros(batch_size, 80 * self.n_group_size, torch.max(n_groups))
                if sigma > 0.0:
                    residual = torch.normal(residual) * sigma

                # map from z sample to data
                num_steps_to_exit = len(self.exit_steps)
                split = num_steps_to_exit * self.n_early_size
                mel = residual[:, split:]
                residual = residual[:, :split]

                for i, flow_step in enumerate(reversed(self.flows)):
                    curr_step = self.n_flows - i - 1
                    mel = flow_step(mel, context_w_spkvec, inverse=True, seq_lens=n_groups)
                    if num_steps_to_exit > 0 and curr_step == self.exit_steps[num_steps_to_exit - 1]:
                        # concatenate the next chunk of z
                        num_steps_to_exit = num_steps_to_exit - 1
                        split = num_steps_to_exit * self.n_early_size
                        residual_to_add = residual[:, split:]
                        residual = residual[:, :split]
                        mel = torch.cat((residual_to_add, mel), 1)

                if self.n_group_size > 1:
                    mel = self.fold(mel)

                return {'mel': mel, 'out_lens': out_lens, 'dur': dur, 'f0': f0, 'energy_avg': energy_avg}

            RadTTSModule.infer = infer

            # Patch module's input_example()
            def input_example(self, max_batch=1, max_dim=400):
                par = next(self.parameters())
                sz = (max_batch, max_dim)
                # sz = (max_batch * max_dim,)
                inp = torch.randint(32, 94, sz, device=par.device, dtype=torch.int64)
                speaker = torch.randint(0, 1, (max_batch,), device=par.device, dtype=torch.int64)
                pitch = torch.randn(sz, device=par.device, dtype=torch.float32) * 0.5
                pace = torch.clamp(torch.randn(sz, device=par.device, dtype=torch.float32) * 0.1 + 1, min=0.01)
                volume = torch.clamp(torch.randn(sz, device=par.device, dtype=torch.float32) * 0.1 + 1, min=0.01)

                # TODO: Shouldn't hardcode but self.tokenizer isn't initlized yet so unsure how
                # to get the pad_id
                pad_id = 94
                inp[inp == pad_id] = pad_id - 1 if pad_id > 0 else pad_id + 1

                lens = []
                for i, _ in enumerate(inp):
                    len_i = random.randint(3, max_dim)
                    lens.append(len_i)
                    inp[i, len_i:] = pad_id
                lens = torch.tensor(lens, device=par.device, dtype=torch.int)

                inputs = {
                    'text': inp,
                    'batch_lengths': lens,
                    # 'batch_lengths': batch_lengths,
                    'speaker_id': speaker,
                    'speaker_id_text': speaker,
                    'speaker_id_attributes': speaker,
                    'pitch': pitch,
                    'pace': pace,
                    'volume': volume,
                }
                return (inputs,)

            RadTTSModule.input_example = input_example
        else:
            # NeMo version >= 1.17.0; can just set the relevant flags
            model.export_config["enable_volume"] = True
            model.export_config["enable_ragged_batches"] = True

    # Patch the model config yaml to add the volume and ragged batch flags
    for art in artifacts:
        if art == 'model_config.yaml':
            model_config = yaml.safe_load(artifacts['model_config.yaml']['content'])
            model_config["export_config"] = {'enable_volume': True, 'enable_ragged_batches': True }
            artifacts['model_config.yaml']['content'] = yaml.dump(model_config).encode()
