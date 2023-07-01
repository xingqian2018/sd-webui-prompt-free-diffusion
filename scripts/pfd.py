import gc
import os
import os.path as osp
from collections import OrderedDict
from copy import copy
from typing import Dict, Optional, Tuple
import importlib
import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing, masking, images
import gradio as gr

from einops import rearrange
from scripts import \
    pfd_global_state, pfd_external_code, \
    pfd_version, pfd_utils, \
    pfd_swin, pfd_seecoder

importlib.reload(pfd_global_state)
importlib.reload(pfd_external_code)
importlib.reload(pfd_utils)
importlib.reload(pfd_swin)
importlib.reload(pfd_seecoder)

from scripts.pfd_ui.pfd_ui_group import PFDUiGroup, UiSeeCoderUnit
from scripts.pfd_logging import pfd_logger
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.images import save_image

import cv2
import numpy as np
import torch
import yaml
import torchvision.transforms as tvtrans

from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

gradio_compat = True
try:
    from distutils.version import LooseVersion
    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass

import tempfile
gradio_tempfile_path = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_tempfile_path, exist_ok=True)

pfd_global_state.update_models()

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def image_dict_from_any(image) -> Optional[Dict[str, np.ndarray]]:
    if image is None:
        return None

    if isinstance(image, (tuple, list)):
        image = {'image': image[0], 'mask': image[1]}
    elif not isinstance(image, dict):
        image = {'image': image, 'mask': None}
    else:  # type(image) is dict
        # copy to enable modifying the dict and prevent response serialization error
        image = dict(image)

    if isinstance(image['image'], str):
        if os.path.exists(image['image']):
            image['image'] = np.array(Image.open(image['image'])).astype('uint8')
        elif image['image']:
            image['image'] = pfd_external_code.to_base64_nparray(image['image'])
        else:
            image['image'] = None            

    # If there is no image, return image with None image and None mask
    if image['image'] is None:
        image['mask'] = None
        return image

    if isinstance(image['mask'], str):
        if os.path.exists(image['mask']):
            image['mask'] = np.array(Image.open(image['mask'])).astype('uint8')
        elif image['mask']:
            image['mask'] = pfd_external_code.to_base64_nparray(image['mask'])
        else:
            image['mask'] = np.zeros_like(image['image'], dtype=np.uint8)
    elif image['mask'] is None:
        image['mask'] = np.zeros_like(image['image'], dtype=np.uint8)

    return image

class Script(scripts.Script):
    model_cache = OrderedDict()

    def __init__(self) -> None:
        super().__init__()
        self.latest_network = None
        self.input_image = None
        self.latest_model_hash = ""
        self.enabled_units = []
        self.detected_map = []
        self.post_processors = []

    def title(self):
        return "prompt-free diffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def get_threshold_block(self, proc):
        pass

    @staticmethod
    def get_default_ui_unit(is_ui=True):
        cls = UiSeeCoderUnit if is_ui else pfd_external_code.SeeCoderUnit
        return cls(enabled=False, model="None")

    def uigroup(self, tabname: str, is_img2img: bool, elem_id_tabname: str):
        group = PFDUiGroup(
            gradio_compat,
            self.infotext_fields,
            Script.get_default_ui_unit(),
        )
        group.render(tabname, elem_id_tabname)
        group.register_callbacks(is_img2img)
        return group.render_and_register_unit(tabname, is_img2img)

    def ui(self, is_img2img):
        """
        this function should create gradio UI elements. 
        See https://gradio.app/docs/#components
        The return value should be an array of all components 
        that are used in processing. Values of those returned 
        components will be passed to run() and process() functions.
        """
        self.infotext_fields = []
        self.paste_field_names = []
        controls = ()
        elem_id_tabname = ("img2img" if is_img2img else "txt2img") + "_prompt_free_diffusion"
        with gr.Group(elem_id=elem_id_tabname):
            with gr.Accordion(f"Prompt-Free Diffusion {pfd_version.version_flag}", open = False, elem_id="prompt_free_diffusion"):
                with gr.Column():
                    controls += (self.uigroup(f"Prompt-Free Diffusion", is_img2img, elem_id_tabname),)
        return controls
    
    @staticmethod
    def clear_model_cache():
        Script.model_cache.clear()
        gc.collect()
        devices.torch_gc()

    @staticmethod
    def load_model(p, unet, model_name):
        if model_name in Script.model_cache:
            pfd_logger.info("Loading model from cache: {}".format(model_name))
            return Script.model_cache[model_name]

        model_path = pfd_global_state.models_paths.get(model_name, None)
        if model_path is None:
            raise RuntimeError(
                "You have not selected any SeeCoder "
                "or {} not found".format(model_name))

        if not osp.isfile(model_path):
            raise ValueError(f"File not found: {model_path}")
        pfd_logger.info("Loading model: {}".format(model_name))

        max_cache_modelnum = shared.opts.data.get("pfd_max_cache_modelnum", 1)
        if max_cache_modelnum>0 and len(Script.model_cache) >= max_cache_modelnum:
            Script.model_cache.popitem(last=False)
            gc.collect()
            devices.torch_gc()
        model_net = Script.build_model(p, unet, model_name)
        if max_cache_modelnum > 0:
            Script.model_cache[model_name] = model_net

        return model_net

    @staticmethod
    def build_model(p, unet, model_name):
        model_path = pfd_global_state.models_paths[model_name]
        model_path = osp.abspath(model_path)
        model_stem = Path(model_path).stem
        model_dir = osp.dirname(model_path)
        script_dir = pfd_global_state.script_dir

        state_dict = pfd_utils.load_state_dict(model_path)
        state_dict = [[ni.replace('ctx.image.', ''), vi] for ni, vi in state_dict.items()]
        state_dict = OrderedDict(state_dict)

        possible_config_filenames = [
            osp.join(model_dir, model_stem + ".yaml"),
            osp.join(script_dir, 'models', model_stem + ".yaml"),
            osp.join(model_dir, model_stem.replace('_fp16', '') + ".yaml"),
            osp.join(script_dir, 'models', model_stem.replace('_fp16', '') + ".yaml"),
            osp.join(model_dir, model_stem.replace('-fp16', '') + ".yaml"),
            osp.join(script_dir, 'models', model_stem.replace('-fp16', '') + ".yaml"),
        ]

        network_config = None
        for possible_config_filename in possible_config_filenames:
            if osp.isfile(possible_config_filename):
                network_config = possible_config_filename
                break

        pfd_logger.info("Loading config: {}".format(network_config))

        with open(network_config, 'r') as f:
            network_cfg = yaml.load(
                f, Loader=yaml.FullLoader)
        
        from scripts.pfd_swin import SwinTransformer
        from scripts.pfd_seecoder import SeeCoder_Decoder
        from scripts.pfd_seecoder import SeeCoder_QueryTransformer
        from scripts.pfd_seecoder import SemanticContextEncoder
        
        seecoder_encoder = SwinTransformer(**network_cfg['seecoder_encoder'])
        seecoder_decoder = SeeCoder_Decoder(**network_cfg['seecoder_decoder'])
        seecoder_qtransformer = SeeCoder_QueryTransformer(
            **network_cfg['seecoder_query_transformer'])
        
        seecoder = SemanticContextEncoder(
            imencoder = seecoder_encoder,
            imdecoder = seecoder_decoder,
            qtransformer = seecoder_qtransformer,)
        
        seecoder.load_state_dict(state_dict, strict=True)
        seecoder.to(p.sd_model.device, dtype=p.sd_model.dtype)
        pfd_logger.info("{} loaded.".format(model_name))
        return seecoder

    @staticmethod
    def get_remote_call(p, attribute, default=None, idx=0, strict=False, force=False):
        if not force and not shared.opts.data.get("control_net_allow_script_control", False):
            return default

        def get_element(obj, strict=False):
            if not isinstance(obj, list):
                return obj if not strict or idx == 0 else None
            elif idx < len(obj):
                return obj[idx]
            else:
                return None

        attribute_value = get_element(getattr(p, attribute, None), strict)
        default_value = get_element(default)
        return attribute_value if attribute_value is not None else default_value

    @staticmethod
    def parse_remote_call(p, unit: pfd_external_code.SeeCoderUnit, idx):
        selector = Script.get_remote_call
        unit.enabled = selector(p, "control_net_enabled", unit.enabled, idx, strict=True)
        unit.model = selector(p, "control_net_model", unit.model, idx)
        unit.weight = selector(p, "control_net_weight", unit.weight, idx)
        unit.image = selector(p, "control_net_image", unit.image, idx)
        return unit

    @staticmethod
    def get_enabled_units(p):
        units = pfd_external_code.get_all_units_in_processing(p)
        enabled_units = []

        if len(units) == 0:
            # fill a null group
            remote_unit = Script.parse_remote_call(p, Script.get_default_ui_unit(), 0)
            if remote_unit.enabled:
                units.append(remote_unit)

        for idx, unit in enumerate(units):
            unit = Script.parse_remote_call(p, unit, idx)
            if not unit.enabled:
                continue

            enabled_units.append(copy(unit))
            if len(units) != 1:
                log_key = f"ControlNet {idx}"
            else:
                log_key = "ControlNet"

            log_value = {
                "model": unit.model,
                "weight": unit.weight,}
            log_value = str(log_value).replace('\'', '').replace('{', '').replace('}', '')

            p.extra_generation_params.update({log_key: log_value})

        return enabled_units

    @staticmethod
    def choose_input_image(
            p: processing.StableDiffusionProcessing, 
            unit: pfd_external_code.SeeCoderUnit,
            idx: int, ) -> np.ndarray:

        image = image_dict_from_any(unit.image)
        if image is None:
            raise ValueError('pfd is enabled but no input image is given')

        # Need to check the image for API compatibility
        if isinstance(image['image'], str):
            from modules.api.api import decode_base64_to_image
            input_image = HWC3(np.asarray(decode_base64_to_image(image['image'])))
        else:
            input_image = HWC3(image['image'])

        assert isinstance(input_image, np.ndarray)
        return input_image
    
    def process(self, p, *args):
        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model

        if self.latest_network is not None:
            # always restore (~0.05s)
            self.latest_network.restore(unet)

        self.enabled_units = Script.get_enabled_units(p)

        if len(self.enabled_units) == 0:
           self.latest_network = None
           return

        # cache stuff
        if self.latest_model_hash != p.sd_model.sd_model_hash:
            Script.clear_model_cache()

        self.latest_model_hash = p.sd_model.sd_model_hash
        image_embedding_all = []
        for idx, unit in enumerate(self.enabled_units):
            input_image = Script.choose_input_image(p, unit, idx)
            if input_image is None: 
                continue
            input_image = tvtrans.ToTensor()(input_image)[None].to(
                p.sd_model.device, dtype=p.sd_model.dtype)
            model_net = self.load_model(p, unet, unit.model)
            with torch.no_grad():
                image_embedding = model_net(input_image)
                image_embedding_all.append(image_embedding[0])

        image_embedding_all = torch.cat(image_embedding_all, dim=0)

        setattr(p, 'hack_seecoder_embedding', image_embedding_all)
        setattr(p, 'hack_seecoder_embedding_weight', unit.weight)
        setattr(p, 'hack_seecoder_cfg_scale', unit.cfg_scale)
        setattr(p, 'hack_seecoder_pf_mode', unit.pf_mode)
        setattr(p, 'hack_seecoder_np_type', unit.np_type)

        from modules import prompt_parser, sd_samplers
        def setup_conds(self):
            sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
            self.step_multiplier = 2 if sampler_config and sampler_config.options.get("second_order", False) else 1

            if self.hack_seecoder_np_type == \
                    pfd_external_code.NegativePromptType.STRICTZERO.value:
                self.uc = prompt_parser.get_learned_conditioning(
                    shared.sd_model, [""], self.steps*self.step_multiplier)
                # self.get_conds_with_caching(
                #     prompt_parser.get_learned_conditioning, 
                #     [""],
                #     self.steps*self.step_multiplier, 
                #     self.cached_uc,
                #     None,)
                new_uc = []
                for ii in self.uc:
                    new_uc_i = []
                    for iii in ii:
                        new_uc_i.append(iii._replace(cond = torch.zeros_like(iii.cond)[0:1]))
                    new_uc.append(new_uc_i)
                self.uc = new_uc

            elif self.hack_seecoder_np_type == \
                    pfd_external_code.NegativePromptType.USINGINPUT.value:
                self.uc = prompt_parser.get_learned_conditioning(
                    shared.sd_model, self.negative_prompts, self.steps*self.step_multiplier)
                # self.uc = self.get_conds_with_caching(
                #     prompt_parser.get_learned_conditioning, 
                #     self.negative_prompts, 
                #     self.steps*self.step_multiplier, 
                #     self.cached_uc)
                
            elif self.hack_seecoder_np_type == \
                    pfd_external_code.NegativePromptType.ANIMENP.value:
                self.uc = prompt_parser.get_learned_conditioning(
                    shared.sd_model, [""], self.steps*self.step_multiplier)
                # self.uc = self.get_conds_with_caching(
                #     prompt_parser.get_learned_conditioning, 
                #     [""], 
                #     self.steps*self.step_multiplier, 
                #     self.cached_uc)
                new_uc = []
                anime_uc = torch.load(pfd_global_state.anime_np_path)
                for ii in self.uc:
                    new_uc_i = []
                    for iii in ii:
                        ucdevice, ucdtype = iii.cond.device, iii.cond.dtype
                        anime_uc = anime_uc.to(
                            device=ucdevice, dtype=ucdtype)
                        new_uc_i.append(iii._replace(cond = anime_uc))
                    new_uc.append(new_uc_i)
                self.uc = new_uc

            if self.hack_seecoder_pf_mode == \
                    pfd_external_code.PromptFreeMode.PFREE.value:
                self.c = prompt_parser.get_multicond_learned_conditioning(
                    shared.sd_model, [""], self.steps*self.step_multiplier)
                # self.c = self.get_conds_with_caching(
                #     prompt_parser.get_multicond_learned_conditioning, 
                #     [""],
                #     self.steps*self.step_multiplier, 
                #     self.cached_c)
                self.cfg_scale = self.hack_seecoder_cfg_scale
                for ii in self.c.batch:
                    for iii in ii:
                        iii.weight = self.hack_seecoder_embedding_weight
                        new_scedules = []
                        for iiii in iii.schedules:
                            cdevice, cdtype = iiii.cond.device, iiii.cond.dtype
                            extra = self.hack_seecoder_embedding.to(
                                device=cdevice, dtype=cdtype)
                            new_scedules.append(iiii._replace(cond = extra))
                        iii.schedules = new_scedules
            elif self.hack_seecoder_pf_mode == \
                    pfd_external_code.PromptFreeMode.PUSE.value:
                self.c = prompt_parser.get_multicond_learned_conditioning(
                    shared.sd_model, self.prompts, self.steps*self.step_multiplier)
                # self.c = self.get_conds_with_caching(
                #     prompt_parser.get_multicond_learned_conditioning, 
                #     self.prompts, 
                #     self.steps*self.step_multiplier, 
                #     self.cached_c)
                for ii in self.c.batch:
                    for iii in ii:
                        new_scedules = []
                        for iiii in iii.schedules:
                            cdevice, cdtype = iiii.cond.device, iiii.cond.dtype
                            extra = self.hack_seecoder_embedding.to(
                                device=cdevice, dtype=cdtype)
                            extra *= self.hack_seecoder_embedding_weight
                            new_scedules.append(
                                iiii._replace(cond = torch.cat([iiii.cond, extra], dim=0)))
                        iii.schedules = new_scedules

        import types
        p._original_setup_conds = types.MethodType(p.setup_conds, p)
        p.setup_conds = types.MethodType(setup_conds, p)

def on_ui_settings():
    section = ('Prompt-Free Diffusion', "Prompt-Free Diffusion")
    shared.opts.add_option("seecoder_config", shared.OptionInfo(
        pfd_global_state.default_conf, "Config file for SeeCoder models", section=section))
    shared.opts.add_option("seecoder_path", shared.OptionInfo(
        "", "Extra path to scan for SeeCoder models (e.g. training output directory)", section=section))
    shared.opts.add_option("seecoder_cache_size", shared.OptionInfo(
        1, "Model cache size (requires restart)", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}, section=section))

# batch_hijack.instance.do_hijack()
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(PFDUiGroup.on_after_component)


