# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import random
import nemo
import torch
import yaml
from nemo.core.neural_types.neural_type import NeuralType
from packaging.version import Version

from nemo2riva.patches.tts.general import batch_from_ragged

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

    # Don't override built-in format
    # export_format is read from schemas, radtts is still currently torchscript in the schema
    format_=kwargs['import_config'].exports[0].export_format
    enable_ragged_batches = (format_ == "ONNX")

    if model.__class__.__name__ == 'RadTTSModel':
        try:
            # For NeMo < 1.17.0
            from nemo.collections.tts.helpers.helpers import regulate_len
        except ModuleNotFoundError as e:
            # For NeMo >= 1.17.0
            from nemo.collections.tts.parts.utils.helpers import regulate_len

        from nemo.collections.tts.modules.radtts import RadTTSModule, adjust_f0, pad_dur, pad_energy_avg_and_f0
        from nemo.core.neural_types.elements import (
            Index,
            LengthsType,
            MelSpectrogramType,
            RegressionValuesType,
            TokenDurationType,
            TokenIndex,
        )
        if nemo_version < Version('1.15.0'):
            # RadTTS came in NeMo 1.11.0 but the export interafce was not stable until 1.15.0
            # RadTTS will require NeMo >= 1.15.0
            raise NotImplementedError(
                "Nemo2riva obtained a RadTTS model, and the installed NeMo version was less than "
                "1.15.0. Please update to nemo_toolkit['all']>=1.15.0"
            )
        elif nemo_version < Version('1.17.0') and not hasattr(model, "export_config"):
            # We need to support NeMo 1.15, and NeMo 1.16. The patches are slightly different for input_example()
            # We need to patch the model using 1.17.0's functions
            using_v15 = (nemo_version < Version('1.16.0'))
            if using_v15:
                logging.warning(
                    "Nemo2riva obtained a RadTTS model and the installed NeMo version was less "
                    "than 1.16.0. Please consider updating to nemo_toolkit['all']>=1.16.0 as "
                    "RadTTS has obtained various bugfixes in 1.16.0 and 1.17.0."
                )

            # Patch model's _prepare_for_export()
            def _prepare_for_export(self, **kwargs):
                self.model.remove_norms()
                super(model.__class__, model)._prepare_for_export(**kwargs)
                tensor_shape = ('T') if enable_ragged_batches else ('B', 'T')
                # Define input_types and output_types as required by export()
                self._input_types = {
                    "text": NeuralType(tensor_shape, TokenIndex()),
                    "batch_lengths": NeuralType(('B')),
                    "speaker_id": NeuralType(('B'), Index()),
                    "speaker_id_text": NeuralType(('B'), Index()),
                    "speaker_id_attributes": NeuralType(('B'), Index()),
                    "pitch": NeuralType(tensor_shape, RegressionValuesType()),
                    "pace": NeuralType(tensor_shape),
                }
                self._output_types = {
                    "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
                    "num_frames": NeuralType(('B'), TokenDurationType()),
                    "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
                }
                self._input_types["volume"] = NeuralType(tensor_shape, optional=True)
                self._output_types["volume_aligned"] = NeuralType(('B', 'T_spec'), RegressionValuesType())

            model.__class__._prepare_for_export = _prepare_for_export

            # Patch model's forward_for_export()
            def forward_for_export(
                self,
                text,
                batch_lengths,
                speaker_id,
                speaker_id_text,
                speaker_id_attributes,
                pitch,
                pace,
                volume,
            ):
                if enable_ragged_batches:
                    text, pitch, pace, volume, lens = batch_from_ragged(
                        text, pitch, pace, batch_lengths=batch_lengths, padding_idx=self.tokenizer_pad, volume=volume,
                    )
                else:
                    lens = batch_lengths.to(dtype=torch.int64)
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
                    # TODO: This is actually 1.14's signature, maybe we can drop this if clause
                    volume_extended, _ = regulate_len(
                        durs_predicted,
                        volume[:, :truncated_length].unsqueeze(-1),
                        pace[:, :truncated_length]
                    )
                volume_extended = volume_extended.squeeze(-1).float()
                return mel.float(), n_frames, dur.float(), volume_extended

            model.__class__.forward_for_export = forward_for_export

            # Patch module's infer()
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

                txt_enc_time_expanded, out_lens = regulate_len(
                    dur,
                    txt_enc.transpose(1, 2),
                    pace,
                    replicate_to_nearest_multiple=True,
                    group_size=self.n_group_size,
                    in_lens=in_lens,
                )
                n_groups = torch.div(out_lens, self.n_group_size, rounding_mode='floor')
                max_out_len = torch.max(out_lens)

                txt_enc_time_expanded.transpose_(1, 2)
                if voiced_mask is None:
                    if self.use_vpred_module:
                        # get logits
                        voiced_mask = self.v_pred_module.infer(txt_enc_time_expanded, spk_vec_attributes, lens=out_lens)
                        try:
                            v_pred_threshold = self.v_pred_threshold
                        except AttributeError as e:
                            v_pred_threshold = 0.5
                        voiced_mask_bool = torch.sigmoid(voiced_mask[:, 0]) > v_pred_threshold
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
                    pitch_shift_spec_len, _ = regulate_len(
                        dur,
                        pitch_shift[:, :txt_len_pad_removed].unsqueeze(-1),
                        pace,
                        replicate_to_nearest_multiple=True,
                        group_size=self.n_group_size,
                        in_lens=in_lens,
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

            # Patch input_example()
            def input_example(self, max_batch=1, max_dim=400):
                par = next(self.parameters())
                sz = (max_batch * max_dim,) if enable_ragged_batches else (max_batch, max_dim)
                inp = torch.randint(32, 94, sz, device=par.device, dtype=torch.int64)
                speaker = torch.randint(0, 1, (max_batch,), device=par.device, dtype=torch.int64)
                pitch = torch.randn(sz, device=par.device, dtype=torch.float32) * 0.5
                pace = torch.clamp(torch.randn(sz, device=par.device, dtype=torch.float32) * 0.1 + 1, min=0.01)
                volume = torch.clamp(torch.randn(sz, device=par.device, dtype=torch.float32) * 0.1 + 1, min=0.01)

                # TODO: Shouldn't hardcode but self.tokenizer isn't initlized yet so unsure how
                # to get the pad_id
                pad_id = 94
                inp[inp == pad_id] = pad_id - 1 if pad_id > 0 else pad_id + 1

                if enable_ragged_batches:
                    batch_lengths = torch.zeros((max_batch + 1), device=device, dtype=torch.int32)
                    left_over_size = sz[0]
                    batch_lengths[0] = 0
                    for i in range(1, max_batch):
                        equal_len = (left_over_size - (max_batch - i)) // (max_batch - i)
                        length = torch.randint(equal_len // 2, equal_len, (1,), device=device, dtype=torch.int32)
                        batch_lengths[i] = length + batch_lengths[i - 1]
                        left_over_size -= length.detach().cpu().numpy()[0]
                    batch_lengths[-1] = left_over_size + batch_lengths[-2]

                    sum = 0
                    index = 1
                    while index < len(batch_lengths):
                        sum += batch_lengths[index] - batch_lengths[index - 1]
                        index += 1
                    assert sum == sz[0], f"sum: {sum}, sz: {sz[0]}, lengths:{batch_lengths}"
                else:
                    batch_lengths = torch.randint(max_dim // 2, max_dim, (max_batch,), device=device, dtype=torch.int32)
                    batch_lengths[0] = max_dim

                inputs = {
                    'text': inp,
                    'batch_lengths': batch_lengths,
                    'speaker_id': speaker,
                    'speaker_id_text': speaker,
                    'speaker_id_attributes': speaker,
                    'pitch': pitch,
                    'pace': pace,
                    'volume': volume,
                }
                return (inputs,)

            if using_v15:
                RadTTSModule.input_example = input_example
            else:
                model.__class__.input_example = input_example
        else:
            # NeMo version >= 1.17.0; can just set the relevant flags
            model.export_config["enable_volume"] = True
            model.export_config["enable_ragged_batches"] = enable_ragged_batches

    # Patch the model config yaml to add the volume and ragged batch flags
    for art in artifacts:
        if art == 'model_config.yaml':
            model_config = yaml.safe_load(artifacts['model_config.yaml']['content'])
            model_config["export_config"] = {'enable_volume': True, 'enable_ragged_batches': enable_ragged_batches }
            artifacts['model_config.yaml']['content'] = yaml.dump(model_config).encode()
