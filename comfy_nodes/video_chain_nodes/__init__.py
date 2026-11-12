"""Promptwood's ComfyUI nodes.

VideoChainQwen21FunInpaint: inpainting with the Qwen-Image 2.1 Fun ControlNet Union. ComfyUI's own
QwenImageDiffsynthControlnet node drives this ControlNet but only exposes a control image and a mask; the image to
keep (the inpaint image) is not an input there. This node hands the source image and the mask (white = draw anew)
to ComfyUI's own QwenImage21FunControlPatch, optionally with a control image (pose, depth, ...).
VideoChainFaceSwap: the face of one person swapped through video frames with FaceFusion's HyperSwap (see
faceswap.py), the identity from up to four pictures of the new person; GPEN face restoration optional.
Copy this folder into ComfyUI/custom_nodes and restart ComfyUI.
"""
import os

import comfy.ldm.qwen_image21.model
import folder_paths
from comfy_extras.nodes_model_patch import QwenImage21FunControlPatch

# The face models' folders (models/insightface, models/facerestore_models): a plain ComfyUI does not know them; other
# face nodes and extra_model_paths.yaml may have added them already, then theirs are kept and ours come after.
for _kind in ("insightface", "facerestore_models"):
    try:
        folder_paths.add_model_folder_path(_kind, os.path.join(folder_paths.models_dir, _kind))
        _paths, _exts = folder_paths.folder_names_and_paths[_kind]
        if _exts:                                   # (no extensions listed: every file counts already)
            folder_paths.folder_names_and_paths[_kind] = (_paths, set(_exts) | {".onnx"})
    except Exception:  # noqa: BLE001  (an older ComfyUI: the face swap then reports the missing folder itself)
        pass


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
        if strength == 0:
            return (model,)
        patched = model.clone()
        control = control_image[:, :, :, :3] if control_image is not None else None
        QwenImage21FunControlPatch(model_patch, vae, control, float(strength),
                                   inpaint_image=inpaint_image[:, :, :, :3], mask=mask).register(patched)
        return (patched,)


def _model_files(folder: str, keyword: str) -> list[str]:
    import folder_paths
    try:
        names = folder_paths.get_filename_list(folder)
    except Exception:  # noqa: BLE001  (the folder is not registered)
        return []
    return sorted(n for n in names if keyword in n.lower() and n.lower().endswith(".onnx"))


def _full(folder: str, name: str) -> str:
    import folder_paths
    return folder_paths.get_full_path_or_raise(folder, name)


_SWAPPERS: dict = {}


class VideoChainFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        swappers = _model_files("insightface", "hyperswap") or ["hyperswap_1b_256.onnx"]
        enhancers = ["none"] + (_model_files("facerestore_models", "gpen") or [])
        return {"required": {"images": ("IMAGE",), "person": ("IMAGE",),
                             "swapper": (swappers,), "enhancer": (enhancers,),
                             "enhance": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05})},
                "optional": {"person_2": ("IMAGE",), "person_3": ("IMAGE",), "person_4": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "swapped")
    FUNCTION = "apply"
    CATEGORY = "image/faceswap"

    def apply(self, images, person, swapper, enhancer, enhance=0.8, person_2=None, person_3=None, person_4=None):
        import numpy as np
        import torch
        import comfy.utils
        from .faceswap import FaceSwapper
        try:
            det = _full("insightface", "models/buffalo_l/det_10g.onnx")
            rec = _full("insightface", "models/buffalo_l/w600k_r50.onnx")
        except Exception as e:  # noqa: BLE001
            raise ValueError("the face models det_10g.onnx and w600k_r50.onnx are needed in "
                             "models/insightface/models/buffalo_l") from e
        enh = None if enhancer == "none" else _full("facerestore_models", enhancer)
        key = (swapper, enh)
        if key not in _SWAPPERS:
            _SWAPPERS.clear()
            _SWAPPERS[key] = FaceSwapper(det, rec, _full("insightface", swapper), enh)
        fs = _SWAPPERS[key]

        def u8(t):
            return (t[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        try:
            pics = [u8(p)[0] for p in (person, person_2, person_3, person_4) if p is not None]
            identity = fs.identity(pics)
            result = images[..., :3].clone().cpu()                 # filled frame by frame (little extra memory)
            bar = comfy.utils.ProgressBar(images.shape[0])

            def frames():
                for k in range(images.shape[0]):
                    yield u8(images[k:k + 1])[0]

            def put(k, f):
                result[k] = torch.from_numpy(f).float() / 255.0
            _none, done = fs.run(frames(), identity, enhance if enh else 0.0,
                                 progress=lambda k, n: bar.update_absolute(k, images.shape[0]), out=put)
        finally:
            _SWAPPERS.clear()                                    # (its models leave the GPU with the job)
        if done == 0:
            raise ValueError("no face was found to swap in this clip")
        return (result, done)


NODE_CLASS_MAPPINGS = {"VideoChainQwen21FunInpaint": VideoChainQwen21FunInpaint,
                       "VideoChainFaceSwap": VideoChainFaceSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoChainQwen21FunInpaint": "Qwen-Image 2.1 Fun Inpaint (Promptwood)",
                              "VideoChainFaceSwap": "Face Swap, HyperSwap (Promptwood)"}
