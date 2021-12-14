import logging

from nemo2riva.schema import check_nemo_version


def bpe_check_inputs_and_version(model, artifacts):
    if model.__class__.__name__ == 'EncDecCTCModelBPE':
        enc_class = model.encoder.__class__.__name__
        if enc_class == "ConformerEncoder":
            logging.info("Checking Nemo version for ConformerEncoder ...")
            check_nemo_version(">=1.6.0rc0")
