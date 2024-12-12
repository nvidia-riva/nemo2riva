[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
# Nemo2Riva

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) is a conversational AI toolkit built for researchers working on automatic speech recognition (ASR), natural language processing (NLP), and text-to-speech synthesis (TTS).
NVIDIA Riva is a GPU-accelerated SDK for building Speech AI applications that are customized for your use case and deliver real-time performance. While TAO Toolkit is the recommended path for typical users of Riva, some developers may prefer to use NeMo because it exposes more of the model and PyTorch internals. Riva supports the ability to import models trained in NeMo.

This repo provides Nemo2Riva: command-line tool to export trained NeMo models saved in .nemo format, to Riva input format (.riva) to be deployed in Riva.

## Export NeMo Models to Riva with NeMo2Riva

Models trained in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) have the format `.nemo`. To use these models in Riva, convert the model checkpoints to `.riva` format for building and deploying with Riva ServiceMaker using the `nemo2riva` tool. The `nemo2riva` tool is currently packaged and available via the Riva Quick Start scripts.

1. Follow the [NeMo installation](https://github.com/NVIDIA/NeMo#installation) instructions to set up a NeMo environment; version 1.10.0 or greater.  From within your NeMo environment, run:

    ```bash

    pip3 install nvidia-pyindex
    pip3 install nemo2riva
    nemo2riva --out /NeMo/<MODEL_NAME>.riva /NeMo/<MODEL_NAME>.nemo
    ```

For additional information and usage, run:

   ```bash

    nemo2riva --help
   ```

    Usage:

    ```
    nemo2riva [-h] [--out OUT] [--validate] [--schema SCHEMA] [--format FORMAT] [--verbose VERBOSE] [--key KEY] source

    When converting NeMo models to Riva `.eff` input format, passing the input `.nemo` as a parameter creates `.riva`.

    If no `--format` is passed, the Riva-preferred format for the supplied model architecture is selected automatically.

    The format is also derived from schema if the `--schema` argument is supplied, or if `nemo2riva` is able to find the schema for this NeMo model
    among known models - there is a set of YAML files in the `nemo2riva/validation_schemas` directory, or you can add your own.

    If the `--key` argument is passed, the model graph in the output EFF file is encrypted with that key.

    positional arguments:

    : source             Source .nemo file

    optional arguments:

    : -h

        --help

        Show this help message and exit

        --out

        OUT

        Location to write resulting Riva EFF input to (default: `None`)

        --validate

        Validate using schemas (default: `False`)

        --schema

        SCHEMA

        Schema file to use for validation (default: `None`)

        --format

        FORMAT

        Force specific export format: `ONNX|TS|CKPT|NEMO|PYTORCH|STATE` (default: `None`)

        --verbose

        VERBOSE

        Verbose level for logging, numeric (default: `None`)

        --key

        KEY

        Encryption key or file (default: `None`)
    ```

## Supported Models 

<table>
  <tr>
    <td><b>Modality</b></td>
    <td><b>Architecture</b></td>
    <td><b>nemo2riva command</b></td>
  </tr>
  <tr>
    <td rowspan="6">Automatic Speech Recognition</td>
    <td>Parakeet-CTC</td>
    <td rowspan="2">nemo2riva --onnx-opset 18 &lt;path to .nemo model&gt;</td>
  </tr>
  <tr>
    <td>Conformer-CTC</td>
  </tr>
  <tr>
    <td>Parakeet-TDT</td>
    <td rowspan="3">nemo2riva --format nemo &lt;path to .nemo model&gt;</td>
  </tr>
    <tr>
    <td>Parakeet-RNNT</td>
  </tr>
      <tr>
    <td>Conformer-RNNT</td>
  </tr>
      <tr>
    <td>Canary</td>
    <td>nemo2riva --format state &lt;path to .nemo model&gt; </td>
  </tr>
  <tr><td colspan="3">Note: Extract CTC/RNNT head from Hybrid ASR using <a href="https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_hybrid_transducer_ctc/helpers/convert_nemo_asr_hybrid_to_ctc.py"> this</a> script before running nemo2riva</td></tr>
    <tr>
    <td rowspan="3">Text to Speech</td>
    <td>HiFiGAN</td>
    <td rowspan="3">nemo2riva &lt;path to .nemo model&gt;</td>
    </tr>
    <tr>
    <td>Fastpitch</td>
    </tr>
    <tr>
    <td>RadTTS</td>
    </tr>
    <tr>
    <td rowspan="2">Voice Activity Detection</td>
    <td>Segment VAD</td>
    <td rowspan="2">nemo2riva --onnx-opset 18 &lt;path to .nemo model&gt;</td>
    <tr>
    <td>FrameVAD</td>
    </tr>
    <tr>
    <td rowspan="2">Punctuation and Capitalization</td>
    <td>Bert-Base</td>
    <td rowspan="2">nemo2riva --onnx-opset 18 &lt;path to .nemo model&gt;</td>
    <tr>
    <td>Bert-Large</td>
    </tr>
    <tr>
    <td>Neural Machine Translation</td>
    <td>Megatron</td>
    <td>nemo2riva &lt;path to .nemo model&gt; </td>
  </tr>
</table>