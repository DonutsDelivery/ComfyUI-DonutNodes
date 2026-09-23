"""Compatibility for saved sampler-local switches; new workflows use the MODEL switch.

No alpha, NAG, edit, SDA or sampling-mode gate. Normalization is implemented by
model-local txtfusion hooks shared with Fusion Control, never by a NAG forward.
"""
from copy import deepcopy
import logging

from .donut_grounding_schedule import DonutSampler as _Base
from .donut_txtfusion_model_guard import attach_model_guard

LOGGER = logging.getLogger(__name__)


class DonutSampler(_Base):
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        schema = deepcopy(super().INPUT_TYPES())
        optional = schema.setdefault("optional", {})
        # Keep both old positions and values readable. New global control is on
        # Fusion Control so the same guarded MODEL also reaches finish stages.
        optional["txtfusion_internal_guard"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Legacy sampler-local switch. Now NAG-independent, including alpha 0, editing and SDA. For all connected stages use Models > Txtfusion RMS guard instead.",
        })
        names = [name for name in folder_paths.get_filename_list("diffusion_models")
                 if name.lower().endswith(".safetensors")]
        optional["txtfusion_reference_checkpoint"] = (["None", *names], {
            "default": "None",
            "tooltip": "Deprecated compatibility field, no file is read. Reference is now captured automatically from the effective checkpoint/merge before adapters, for both native and bypass execution.",
        })
        return schema

    # Explicit parent signature instead of reflected argument binding: the
    # registry scanner misreads reflection here as socket binding (3.0.29
    # through 3.0.33 were Flagged for exactly that). Keyword-only and
    # **nag_options values are forwarded unchanged via **kwargs; grounded
    # keyword defaults match the parent's own defaults so omitted values keep
    # their meaning.
    def sample(
        self, model, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step,
        sampler_name, scheduler, positive, negative, latent_image, denoise,
        mode="simple", cfg_curve="linear", add_noise="enable", start_at_step=0,
        end_at_step=10000, return_with_leftover_noise="disable",
        randomize_seed_per_model="enable", switch_at_step_1=10,
        switch_at_step_2=15, model_2=None, model_3=None, edit_mode=False,
        source_image=None, vae=None, clip=None, edit_prompt="",
        edit_negative_prompt="", grounding_px=768, edit_model=None,
        turbo_mode=False, source_image_b=None, edit_inpaint=None,
        sda_enabled=False, sda_strength=1.0, *args,
        txtfusion_internal_guard=False, txtfusion_reference_checkpoint="None",
        **kwargs,
    ):
        if type(txtfusion_internal_guard) is not bool:
            raise ValueError("txtfusion_internal_guard must be a boolean")
        forwarded = dict(
            seed=seed, steps=steps, cfg_start=cfg_start, cfg_halfway=cfg_halfway,
            cfg_end=cfg_end, halfway_step=halfway_step, sampler_name=sampler_name,
            scheduler=scheduler, positive=positive, negative=negative,
            latent_image=latent_image, denoise=denoise, mode=mode,
            cfg_curve=cfg_curve, add_noise=add_noise, start_at_step=start_at_step,
            end_at_step=end_at_step, return_with_leftover_noise=return_with_leftover_noise,
            randomize_seed_per_model=randomize_seed_per_model,
            switch_at_step_1=switch_at_step_1, switch_at_step_2=switch_at_step_2,
            edit_mode=edit_mode, source_image=source_image, vae=vae, clip=clip,
            edit_prompt=edit_prompt, edit_negative_prompt=edit_negative_prompt,
            grounding_px=grounding_px, edit_model=edit_model,
            turbo_mode=turbo_mode, source_image_b=source_image_b,
            edit_inpaint=edit_inpaint, sda_enabled=sda_enabled,
            sda_strength=sda_strength, **kwargs,
        )
        if not txtfusion_internal_guard:
            # Do not remove a model-wide guard selected upstream.
            return super().sample(model=model, *args, **forwarded)
        if txtfusion_reference_checkpoint != "None":
            LOGGER.warning("[Donut txtfusion RMS] legacy reference filename is not used; "
                           "the effective model/merge supplies the checkpoint reference")
        if model_2 is not None:
            model_2 = attach_model_guard(model_2)
        if model_3 is not None:
            model_3 = attach_model_guard(model_3)
        return super().sample(model=attach_model_guard(model), model_2=model_2,
                              model_3=model_3, *args, **forwarded)


NODE_CLASS_MAPPINGS = {"DonutSampler": DonutSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSampler": "DonutSampler"}
