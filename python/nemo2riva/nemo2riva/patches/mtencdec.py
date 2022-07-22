import yaml


def change_tokenizer_names(model, artifacts):
    if model.__class__.__name__ == 'MTEncDecModel':

        conf = yaml.safe_load(artifacts['model_config.yaml']['content'])

        enctok = conf['encoder_tokenizer']['tokenizer_model']
        if enctok not in artifacts:
            enctok = enctok.replace("nemo:", "")
        artifacts['encoder_tokenizer.model'] = artifacts.pop(enctok)
        conf['encoder_tokenizer']['tokenizer_model'] = 'encoder_tokenizer.model'

        dectok = conf['decoder_tokenizer']['tokenizer_model']
        if dectok not in artifacts:
            dectok = dectok.replace("nemo:", "")
        artifacts['decoder_tokenizer.model'] = artifacts.pop(dectok)
        conf['decoder_tokenizer']['tokenizer_model'] = 'decoder_tokenizer.model'

        artifacts['model_config.yaml']['content'] = yaml.safe_dump(conf, encoding=('utf-8'))
