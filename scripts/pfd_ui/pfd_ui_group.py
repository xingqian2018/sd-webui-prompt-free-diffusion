import gradio as gr
import functools
from typing import List, Optional, Union, Dict, Callable
import numpy as np
import base64

from scripts import pfd_global_state
from scripts import pfd_external_code
from scripts.pfd_logging import pfd_logger
from modules import shared
from modules.ui_components import FormRow

class ToolButton(gr.Button, gr.components.FormComponent):
    def __init__(self, **kwargs):
        super().__init__(variant="tool", elem_classes=["pfd-toolbutton"], **kwargs)

    def get_block_name(self):
        return "button"

class UiSeeCoderUnit(pfd_external_code.SeeCoderUnit):
    def __init__(self, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.is_ui = True

class PFDUiGroup(object):
    # Note: Change symbol hints mapping in `javascript/hints.js` when you change the symbol values.
    refresh_symbol = "\U0001f504"  # 🔄

    txt2img_submit_button = None
    img2img_submit_button = None
    txt2img_w_slider = None
    txt2img_h_slider = None
    img2img_w_slider = None
    img2img_h_slider = None

    def __init__(
            self,
            gradio_compat: bool,
            infotext_fields: List[str],
            default_unit: pfd_external_code.SeeCoderUnit,):

        self.gradio_compat = gradio_compat
        self.infotext_fields = infotext_fields
        self.default_unit = default_unit
        self.model = None

    def render(self, tabname: str, elem_id_tabname: str) -> None:
        """
        The pure HTML structure of a single UI. Calling this
        function will populate `self` with all gradio element declared
        in local scope.
        Args:
            tabname:
            elem_id_tabname:
        Returns:
            None
        """
        with gr.Tabs():
            with gr.Tab(label="RefImageMode") as self.upload_tab:
                with gr.Row(elem_classes=["cnet-image-row"]).style(equal_height=True):
                    with gr.Group(elem_classes=["cnet-input-image-group"]):
                        self.input_image = gr.Image(
                            source="upload",
                            brush_radius=20,
                            mirror_webcam=False,
                            type="numpy",
                            elem_id=f"{elem_id_tabname}_{tabname}_input_image",
                            elem_classes=["cnet-image"],
                        )

        with FormRow(
                elem_classes=["checkboxes-row", "seecoder_main_options"],
                variant="compact",):
            self.enabled = gr.Checkbox(
                label="Enable",
                value=self.default_unit.enabled,
                elem_id=f"{elem_id_tabname}_{tabname}_seecoder_enable_checkbox",
                elem_classes=['cnet-unit-enabled'],
            )

        with gr.Row(elem_classes="seecoder"):
            self.model = gr.Dropdown(
                list(pfd_global_state.models_paths.keys()),
                label=f"SeeCoder",
                value=self.default_unit.model,
                elem_id=f"{elem_id_tabname}_{tabname}_seecoder_dropdown",)
            self.refresh_models = ToolButton(
                value=PFDUiGroup.refresh_symbol,
                elem_id=f"{elem_id_tabname}_{tabname}_seecoder_refresh_models",)

        with gr.Row(elem_classes="seecoder_cfg_and_weight"):
            self.cfg_scale = gr.Slider(
                label=f"SeeCoder CFG Scale",
                value=self.default_unit.cfg_scale,
                minimum=1.0,
                maximum=5.0,
                step=0.05,
                elem_id=f"{elem_id_tabname}_{tabname}_seecoder_cfg_scale_slider",
                elem_classes="c_seecoder_cfg_scale_slider",)
            self.weight = gr.Slider(
                label=f"SeeCoder Weight",
                value=self.default_unit.weight,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                elem_id=f"{elem_id_tabname}_{tabname}_seecoder_weight_slider",
                elem_classes="seecoder_weight_slider",)

        with gr.Row(elem_classes="seecoder_mode"):
            self.pf_mode = gr.Radio(
                choices=[e.value for e in pfd_external_code.PromptFreeMode],
                value=self.default_unit.pf_mode.value,
                label="Prompt Free Mode",
                elem_id=f"{elem_id_tabname}_{tabname}_seecoder_prompt_free_mode_radio",
                elem_classes="seecoder_prompt_free_mode_radio",)

            self.np_type = gr.Radio(
                choices=[e.value for e in pfd_external_code.NegativePromptType],
                value=self.default_unit.np_type.value,
                label="Negative Prompt Type",
                elem_id=f"{elem_id_tabname}_{tabname}_seecoder_negative_prompt_type_radio",
                elem_classes="seecoder_negative_prompt_type_radio",)

    def register_refresh_all_models(self):
        def refresh_all_models(*inputs):
            pfd_global_state.update_models()
            dd = inputs[0]
            selected = dd if dd in pfd_global_state.models_paths else "None"
            return gr.Dropdown.update(
                value=selected, choices=list(pfd_global_state.models_paths.keys()))

        self.refresh_models.click(refresh_all_models, self.model, self.model)

    def register_callbacks(self, is_img2img: bool):
        self.register_refresh_all_models()

    def register_modules(
            self, tabname, enabled, cfg_scale, model, weight, pf_mode, np_type):
        self.infotext_fields.extend([
            (enabled, f"{tabname} Enabled"),
            (model, f"{tabname} Model"),
            (cfg_scale, f"{tabname} Cfg_Scale"),
            (weight, f"{tabname} Weight"),
            (pf_mode, f"{tabname} Prompt_Free_Mode"), 
            (np_type, f"{tabname} Negative_Prompt_Type"), ])

    def render_and_register_unit(self, tabname: str, is_img2img: bool):
        unit_args = (
            self.enabled,
            self.model,
            self.cfg_scale,
            self.weight,
            self.input_image,
            self.pf_mode,
            self.np_type,)

        self.register_modules(
            tabname,
            self.enabled,
            self.cfg_scale,
            self.model,
            self.weight,
            self.pf_mode,
            self.np_type,)

        unit = gr.State(self.default_unit)
        for comp in unit_args:
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)
            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)

            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=UiSeeCoderUnit, inputs=list(unit_args), outputs=unit)

        (
            PFDUiGroup.img2img_submit_button
            if is_img2img
            else PFDUiGroup.txt2img_submit_button
        ).click(
            fn=UiSeeCoderUnit,
            inputs=list(unit_args),
            outputs=unit,
            queue=False,
        )

        return unit

    @staticmethod
    def on_after_component(component, **_kwargs):
        elem_id = getattr(component, "elem_id", None)
        if elem_id == "txt2img_generate":
            PFDUiGroup.txt2img_submit_button = component
            return
        if elem_id == "img2img_generate":
            PFDUiGroup.img2img_submit_button = component
            return
