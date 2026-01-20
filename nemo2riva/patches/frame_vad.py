from nemo.core.neural_types.neural_type import NeuralType
from typing import Any, Dict, Optional
from nemo.core.neural_types.elements import LogitsType
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core.utils.neural_type_utils import get_io_names

def patch_output_name(model, artifacts, **kwargs):
    if model.__class__.__name__ == "EncDecFrameClassificationModel":
        @property
        def output_names(self):
            return ["logits"]   
        model.__class__.output_names = output_names   
        






