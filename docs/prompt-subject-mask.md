# Select Reference B with a mask prompt

In **Edit Studio → Smart subject mask · B**, choose **Prompt selection**.
Enter an object in **Mask prompt**, such as `hat`, `shirt`, or `person`, then
click **Select from prompt**. The queued SAM3.1 job generates a preview. Inspect
it and use the existing paint/erase editor to refine it. The selected mask is
saved with the workflow and applied before cropping Reference B.

White keeps the described object; black replaces the rest with the chosen
background. This masks **Reference B**, not the area to repaint in A. Use A's
Paint / outpaint editor to choose where generation changes the base image.

Lower **Detection threshold** if the object is missed. Simple object names are
the most predictable; descriptions are not a guarantee of relational reasoning
such as selecting a particular person among several similar people. No match
reports an error instead of silently using the whole image. Selecting again
after changing the prompt generates a new mask. Prompt, threshold, source pixels
and model identity all participate in the inference cache.

**Auto subject** remains the BiRefNet whole-foreground mode. **Saved mask** keeps
your reviewed selection; choose Prompt selection again to generate another one.
Prompt selection left active during generation recomputes when inputs change.

Requires ComfyUI's native `SAM3_Detect` support and
`models/checkpoints/sam3.1_multiplex_fp16.safetensors` (1.75 GB). Both Download
missing and the standalone installer include the checkpoint from the shared
catalog, with its pinned upstream revision and SHA-256. Registry builds provide
manual model links and the separate installer link. Runtime masking performs no
automatic model download.
