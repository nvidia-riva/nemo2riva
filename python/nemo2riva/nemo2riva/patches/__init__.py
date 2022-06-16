from .ctc_bpe import bpe_check_inputs_and_version
from .fastpitch import generate_vocab_mapping, patch_volume
from .mtencdec import change_tokenizer_names

patches = {
    "EncDecCTCModelBPE": [bpe_check_inputs_and_version],
    "MTEncDecModel": [change_tokenizer_names],
    "FastPitchModel": [generate_vocab_mapping, patch_volume],
}
