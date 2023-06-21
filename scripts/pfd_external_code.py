from enum import Enum
from typing import List, Any, Optional, Union, Tuple, Dict
import numpy as np
from modules import scripts, processing, shared
from scripts.pfd_logging import pfd_logger
from modules.api import api

def get_api_version() -> int:
    return 1

InputImage = Union[np.ndarray, str]
InputImage = Union[Dict[str, InputImage], Tuple[InputImage, InputImage], InputImage]

class PromptFreeMode(Enum):
    PFREE = "Text prompt free"
    PUSE = "Still using text prompt"

class NegativePromptType(Enum):
    STRICTZERO = "Strict zero"
    USINGINPUT = "Using input"
    ANIMENP = "Default anime NP"

class SeeCoderUnit:
    def __init__(
            self,
            enabled: bool=True,
            model: Optional[str]=None,
            cfg_scale: float=2.0,
            weight: float=1.0,
            image: Optional[InputImage]=None,
            pf_mode: Union[PromptFreeMode, int, str] = PromptFreeMode.PFREE,
            np_type: Union[NegativePromptType, int, str] = NegativePromptType.STRICTZERO,
            **_kwargs):
        self.enabled = enabled
        self.model = model
        self.cfg_scale = cfg_scale
        self.weight = weight
        self.image = image
        self.pf_mode = pf_mode
        self.np_type = np_type

    def __eq__(self, other):
        if not isinstance(other, SeeCoderUnit):
            return False
        return vars(self) == vars(other)

def to_base64_nparray(encoding: str):
    return np.array(api.decode_base64_to_image(encoding)).astype('uint8')

def get_all_units_in_processing(p: processing.StableDiffusionProcessing) -> List[SeeCoderUnit]:
    return get_all_units(p.scripts, p.script_args)

def get_all_units(script_runner: scripts.ScriptRunner, script_args: List[Any]) -> List[SeeCoderUnit]:
    script = find_script(script_runner)
    if script:
        return get_all_units_from(script_args[script.args_from:script.args_to])
    return []

def find_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    if script_runner is None:
        return None
    for script in script_runner.alwayson_scripts:
        if is_script(script):
            return script

def is_script(script: scripts.Script) -> bool:
    return script.title().lower() == 'prompt-free diffusion'

def get_all_units_from(script_args: List[Any]) -> List[SeeCoderUnit]:
    units = []
    i = 0
    while i < len(script_args):
        if script_args[i] is not None:
            units.append(to_processing_unit(script_args[i]))
        i += 1
    return units

def to_processing_unit(unit: Union[Dict[str, Any], SeeCoderUnit]) -> SeeCoderUnit:
    ext_compat_keys = {
        'input_image': 'image', }

    if isinstance(unit, dict):
        unit = {ext_compat_keys.get(k, k): v for k, v in unit.items()}
        mask = None
        if 'mask' in unit:
            mask = unit['mask']
            del unit['mask']
        if 'image' in unit and not isinstance(unit['image'], dict):
            unit['image'] = {'image': unit['image'], 'mask': mask} \
                if mask is not None else unit['image'] if unit['image'] else None
        unit = SeeCoderUnit(**unit)
    return unit
