"""Promptwood's ComfyUI nodes.

VideoChainQwen21FunInpaint: inpainting with the Qwen-Image 2.1 Fun ControlNet Union. ComfyUI's own
QwenImageDiffsynthControlnet node drives this ControlNet but only exposes a control image and a mask; the image to
keep (the inpaint image) is not an input there. This node hands the source image and the mask (white = draw anew)
to ComfyUI's own QwenImage21FunControlPatch, optionally with a control image (pose, depth, ...).
Copy this folder into ComfyUI/custom_nodes and restart ComfyUI.
"""
import comfy.ldm.qwen_image21.model
from comfy_extras.nodes_model_patch import QwenImage21FunControlPatch


class VideoChainQwen21FunInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "model_patch": ("MODEL_PATCH",),
                             "vae": ("VAE",),
                             "inpaint_image": ("IMAGE",),
                             "mask": ("MASK",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})},
                "optional": {"control_image": ("IMAGE",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/patch/qwen"

    def apply(self, model, model_patch, vae, inpaint_image, mask, strength=1.0, control_image=None):
        if not isinstance(model_patch.model, comfy.ldm.qwen_image21.model.QwenImage21FunControl):
            raise ValueError("this model patch is not the Qwen-Image 2.1 Fun ControlNet")
        patched = model.clone()
        control = control_image[:, :, :, :3] if control_image is not None else None
        QwenImage21FunControlPatch(model_patch, vae, control, float(strength),
                                   inpaint_image=inpaint_image[:, :, :, :3], mask=mask).register(patched)
        return (patched,)


NODE_CLASS_MAPPINGS = {"VideoChainQwen21FunInpaint": VideoChainQwen21FunInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoChainQwen21FunInpaint": "Qwen-Image 2.1 Fun Inpaint (Promptwood)"}
