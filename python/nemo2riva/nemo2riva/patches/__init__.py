from .ctc_bpe import bpe_check_inputs_and_version
from .fastpitch import generate_vocab_mapping

patches = {
    "FastPitchModel": [generate_vocab_mapping],
    "EncDecCTCModelBPE": [bpe_check_inputs_and_version],
}
