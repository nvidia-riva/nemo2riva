import logging

from nemo.utils import model_utils


def generate_vocab_mapping(model, artifacts):
    # TODO Hack to add labels from FastPitch to .riva since that file is not inside the .nemo
    # Task tracked at https://jirasw.nvidia.com/browse/JARS-1169
    if model.__class__.__name__ == 'FastPitchModel' and hasattr(model, 'vocab'):
        logging.info("Adding mapping.txt for FastPitchModel instance to output file")
        labels = model.vocab.labels
        mapping = []
        for idx, token in enumerate(labels):
            if not str.islower(token) and str.isalnum(token):
                # token is ARPABET token, need to be prepended with @
                token = '@' + token
            mapping.append("{} {}".format(idx, token))
            if str.islower(token) and str.isalnum(token):
                # normal lowercase token, we want to create uppercase variant too
                # since nemo preprocessing includes a .tolower
                mapping.append("{} {}".format(idx, token.upper()))

        mapping_txt = "\n".join(mapping)

        content = {
            "description": "mapping file for FastPitch",
            "conf_path": "./mapping.txt",
            "path_type": model_utils.ArtifactPathType.TAR_PATH,
            "nemo_artifact": True,
            "content": mapping_txt,
        }
        artifacts["mapping.txt"] = content
