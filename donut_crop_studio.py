"""Per-image geometry owned by the existing Edit/Reference Studio modules.

New inputs append after the existing masking inputs. The legacy route is
forwarded unchanged. Model/VAE reference resolutions are deliberately not exposed.
"""
from copy import deepcopy
import json

from .donut_inpaint import prepare_outpaint
from . import DonutEditStudio as studio
from . import donut_reference_mask as subjects
from .donut_reference_geometry import (
    LEGACY, INDEPENDENT, FIT_KEY, crop_pixels, crop_fit_image, output_dimensions,
    pil_tensor, fit_mask,
)

_Base = subjects.DonutSubjectMaskStudio



def crop_input_types(edit=True):
    result = {
        "geometry_mode": ([LEGACY, INDEPENDENT], {"default": LEGACY,
            "tooltip": "Legacy preserves saved output-linked crops. Independent crops keeps each source selection when output dimensions change."}),
        "crop_data_a": ("STRING", {"default": "", "dynamicPrompts": False}),
        "crop_data_b": ("STRING", {"default": "", "dynamicPrompts": False}),
    }
    if edit:
        result["output_canvas"] = (["Follow A crop", "Independent output"], {"default": "Follow A crop",
            "tooltip": "Only used with Independent crops while editing. Follow A uses its crop aspect; independent output fits A/B into the selected global canvas."})
    return result


def isolated_b(name, source, options):
    mode = options.get("mask_b_mode", "Off")
    if mode == "Off":
        return source, None, 0.5
    if source.width * source.height > subjects.MAX_PIXELS:
        raise ValueError("Reference B is too large for Smart Mask; use an image under 32 megapixels.")
    if mode == "Auto subject":
        record, mask = subjects.auto_mask(name, source, options.get("mask_b_model", subjects.MODEL_NAME))
    elif mode == "Prompt selection":
        record, mask = subjects.prompt_mask(name, source, options.get("mask_b_prompt", ""), options.get("mask_b_threshold", 0.5))
    elif mode == "Saved mask":
        try:
            record = json.loads(options.get("mask_b_data", ""))
        except (ValueError, TypeError):
            raise ValueError("No valid saved B mask. Auto select or paint a mask first.") from None
        mask = subjects.load_mask(record, name, source)
    elif mode == "External mask":
        mask = options.get("mask_b")
        if mask is None:
            raise ValueError("Connect an original-B-aligned MASK or choose Auto subject.")
        record = subjects.store_mask(name, source, mask)
        mask = subjects.load_mask(record, name, source)
    else:
        raise ValueError("Unknown Reference B mask mode.")
    background = options.get("mask_b_background", "Neutral gray")
    source = subjects.composite(source, mask, options.get("mask_b_grow", 0),
                                options.get("mask_b_feather", 0), background)
    return source, record, subjects.BACKGROUNDS[background]


class DonutCropEditStudio(_Base):
    @classmethod
    def INPUT_TYPES(cls):
        result = deepcopy(super().INPUT_TYPES())
        result.setdefault("optional", {}).update(crop_input_types())
        result["optional"].update(subjects.prompt_mask_inputs())
        return result

    def prepare(
        self,
        enabled,
        image_a,
        image_b,
        use_reference_b,
        prompt,
        resolution_mode,
        aspect_ratio,
        megapixels,
        width,
        height,
        multiple,
        grounding_px,
        lora_name,
        lora_strength,
        crop_a_x=0.5,
        crop_a_y=0.5,
        crop_b_x=0.5,
        crop_b_y=0.5,
        model=None,
        text_seed=0,
        inpaint_enabled=False,
        mask_data='',
        mask_feather=8,
        grounding_schedule='constant',
        grounding_start_px=512,
        grounding_end_px=1088,
        *,
        mask_b_mode="Off",
        mask_b_model=subjects.MODEL_NAME,
        mask_b_data="",
        mask_b_grow=0,
        mask_b_feather=0,
        mask_b_background="Neutral gray",
        mask_b=None,
        mask_b_prompt="",
        mask_b_threshold=0.5,
        geometry_mode=LEGACY,
        crop_data_a="",
        crop_data_b="",
        output_canvas="Follow A crop",
    ):
        values = dict(
            enabled=enabled,
            image_a=image_a,
            image_b=image_b,
            use_reference_b=use_reference_b,
            prompt=prompt,
            resolution_mode=resolution_mode,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            width=width,
            height=height,
            multiple=multiple,
            grounding_px=grounding_px,
            lora_name=lora_name,
            lora_strength=lora_strength,
            crop_a_x=crop_a_x,
            crop_a_y=crop_a_y,
            crop_b_x=crop_b_x,
            crop_b_y=crop_b_y,
            model=model,
            text_seed=text_seed,
            inpaint_enabled=inpaint_enabled,
            mask_data=mask_data,
            mask_feather=mask_feather,
            grounding_schedule=grounding_schedule,
            grounding_start_px=grounding_start_px,
            grounding_end_px=grounding_end_px,
        )
        mask_options = dict(
            mask_b_mode=mask_b_mode,
            mask_b_model=mask_b_model,
            mask_b_data=mask_b_data,
            mask_b_grow=mask_b_grow,
            mask_b_feather=mask_b_feather,
            mask_b_background=mask_b_background,
            mask_b=mask_b,
            mask_b_prompt=mask_b_prompt,
            mask_b_threshold=mask_b_threshold,
        )
        if geometry_mode == LEGACY:
            return super().prepare(**values, **mask_options)
        if geometry_mode != INDEPENDENT:
            raise ValueError("Unknown reference geometry mode.")
        # No image, mask, model or crop parsing while editing is off.
        if not values["enabled"]:
            return super().prepare(**values, **mask_options)
        a = studio._open_reference(values["image_a"])
        need_b = values["image_b"] and (values["use_reference_b"] or values["aspect_ratio"] == "Auto · Reference B")
        b = studio._open_reference(values["image_b"]) if need_b else None
        box_a = crop_pixels(crop_data_a, values["image_a"], a.size)
        box_b = crop_pixels(crop_data_b, values["image_b"], b.size) if b is not None else None
        size = output_dimensions({**values, "geometry_mode": geometry_mode, "output_canvas": output_canvas},
                                 a.size, box_a, b.size if b else None, box_b)
        # Delegate prompt/LoRA/grounding preparation to the original owner. Do
        # not run its legacy inpaint crop: that could discard a valid new mask.
        legacy = {**values, "resolution_mode": "Custom", "width": size[0], "height": size[1],
                  "inpaint_enabled": False}
        result = list(studio.DonutEditStudio.prepare(self, **legacy))
        result[0], fit_a = crop_fit_image(a, box_a, size)
        result[3:5] = size
        record = None
        if values["use_reference_b"]:
            if b is None:
                raise ValueError("Reference B is required when Use B is enabled.")
            b, record, fill = isolated_b(values["image_b"], b, mask_options)
            result[1], _ = crop_fit_image(b, box_b, size, background=fill)
        if values["inpaint_enabled"]:
            outpaint = prepare_outpaint(a, values["mask_data"], values["image_a"], size, values["mask_feather"])
            if outpaint is not None:
                result[0], result[8] = outpaint["image"], outpaint
            else:
                content_mask = studio.rasterize_mask(values["mask_data"], values["image_a"],
                    a.size, box_a, fit_a.content, 0)
                result[8] = {"image": result[0], "mask": fit_mask(content_mask, fit_a, values["mask_feather"])}
        edit_model = result[6]
        if not hasattr(edit_model, "clone") or not isinstance(getattr(edit_model, "model_options", None), dict):
            raise TypeError("Independent crops require a ComfyUI model patcher.")
        edit_model = edit_model.clone()
        edit_model.model_options[FIT_KEY] = True
        result[6] = edit_model
        ui = {"donut_crop_geometry": [{"width": size[0], "height": size[1],
                                      "image_a": values["image_a"], "box_a": list(box_a),
                                      "image_b": values["image_b"], "box_b": list(box_b) if box_b else None}]}
        if record:
            ui["donut_subject_mask"] = [record]
        return {"result": tuple(result), "ui": ui}


class DonutCropReferenceStudio(studio.DonutReferenceStudio):
    @classmethod
    def INPUT_TYPES(cls):
        result = deepcopy(super().INPUT_TYPES())
        result.setdefault("optional", {}).update(crop_input_types(edit=False))
        return result

    @classmethod
    def IS_CHANGED(cls, enabled=False, image_a="", image_b="", use_reference_b=False,
                   edit_active=False, **kwargs):
        return super().IS_CHANGED(enabled, image_a, image_b, use_reference_b, edit_active)

    def prepare(self, enabled=False, image_a="", image_b="", use_reference_b=False,
                edit_active=False, geometry_mode=LEGACY, crop_data_a="", crop_data_b=""):
        if geometry_mode == LEGACY or not enabled or edit_active:
            return super().prepare(enabled, image_a, image_b, use_reference_b, edit_active)
        if geometry_mode != INDEPENDENT:
            raise ValueError("Unknown reference geometry mode.")
        def load(name, data):
            if not name:
                raise ValueError("Add the reference image or turn guidance off.")
            source = studio._open_reference(name)
            return pil_tensor(source.crop(crop_pixels(data, name, source.size)))
        # Native reference guidance accepts each image independently. There is
        # no output canvas here, so do not introduce artificial padding/resize.
        return load(image_a, crop_data_a), load(image_b, crop_data_b) if use_reference_b else None, True


NODE_CLASS_MAPPINGS = {"DonutEditStudio": DonutCropEditStudio, "DonutReferenceStudio": DonutCropReferenceStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutEditStudio": "Donut Edit Studio", "DonutReferenceStudio": "Donut Reference Guidance"}
