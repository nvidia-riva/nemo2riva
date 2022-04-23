import yaml


def change_tokenizer_names(model, artifacts, **kwargs):
    if model.__class__.__name__ == 'MTEncDecModel':

        conf = yaml.safe_load(artifacts['model_config.yaml']['content'])

        enctok = conf['encoder_tokenizer']['tokenizer_model']
        if enctok not in artifacts:
            enctok = enctok.strip("nemo:")
        artifacts['encoder_tokenizer.model'] = artifacts.pop(enctok)
        conf['encoder_tokenizer']['tokenizer_model'] = 'encoder_tokenizer.model'

        dectok = conf['decoder_tokenizer']['tokenizer_model']
        if dectok not in artifacts:
            dectok = dectok.strip("nemo:")
        artifacts['decoder_tokenizer.model'] = artifacts.pop(dectok)
        conf['decoder_tokenizer']['tokenizer_model'] = 'decoder_tokenizer.model'

        artifacts['model_config.yaml']['content'] = yaml.safe_dump(conf, encoding=('utf-8'))

        if 'export_subnet' in kwargs:
            if kwargs['export_subnet'] == 'encoder':
                del artifacts['decoder_tokenizer.model']
            elif kwargs['export_subnet'] == 'decoder':
                del artifacts['encoder_tokenizer.model']
            else:
                del artifacts['decoder_tokenizer.model']
                del artifacts['encoder_tokenizer.model']
