# Original Balanced baseline

Extracted from /home/user/ComfyUI_00922_.png.

Balanced-original-compatible-api.json is an API-format workflow containing the original main sampling and VAE decode path, with a new SaveImage output. Original prompt and UI workflow are preserved separately.

Compatibility changes: rename the legacy Balanced label to Balanced; explicitly use Fusion only and Advanced mode; supply the new LoRA safety inputs with safety/budgeting Off; explicitly disable NAG and Turbo mode, which were absent from the historical sampler inputs. Remove branches not required for main generation, including detailers, upscaling, image editing, missing conversion nodes and VRAM purge nodes.

Preserved: original sampling seed, prompt/wildcard graph, model and LoRA selections, dimensions, Balanced numeric inputs, sampler, scheduler and eight steps at CFG 1.

Validation: all required inputs and enum choices checked against the running ComfyUI object_info. No generation was queued. This is a main-generation baseline, not a reproduction of the PNG's later upscale/detailer stages. Wildcard files, random-number nodes and model file contents may have changed since the original run, so exact pixel reproduction is not guaranteed.
