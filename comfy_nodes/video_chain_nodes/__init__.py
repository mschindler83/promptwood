"""Promptwood's ComfyUI nodes.

VideoChainQwen21FunInpaint: inpainting with the Qwen-Image 2.1 Fun ControlNet Union. ComfyUI's own
QwenImageDiffsynthControlnet node drives this ControlNet but only exposes a control image and a mask; the image to
keep (the inpaint image) is not an input there. This node hands the source image and the mask (white = draw anew)
to ComfyUI's own QwenImage21FunControlPatch, optionally with a control image (pose, depth, ...).
VideoChainFaceSwap: the face of one person swapped through video frames with FaceFusion's HyperSwap (see
faceswap.py), the identity from up to four pictures of the new person; GPEN face restoration optional.
VideoChainLanczosScale: ComfyUI's lanczos resize with the same frames, on all CPU cores (see scale.py).
VideoChainH3LoRA: a MiniMax-H3 LoRA as ComfyUI's loader applies it; on the pruned H3 files also the adaln part of a
LoRA trained on the unpruned model, added at run time (see h3_lora.py).
VideoChainH3PinNoise: the noise level MiniMax H3 conditions its pinned frames and reference pictures at (see
pin_noise.py).
Copy this folder into ComfyUI/custom_nodes and restart ComfyUI.
"""
import logging
import os

try:                                     # ComfyUI's Qwen-Image 2.1 internals: not in every version; only this node needs them
    import comfy.ldm.qwen_image21.model
    from comfy_extras.nodes_model_patch import QwenImage21FunControlPatch
    HAS_QWEN21 = True
except ImportError:
    HAS_QWEN21 = False

log = logging.getLogger(__name__)


def _face_folders() -> None:
    """The face models' folders (models/insightface, models/facerestore_models): a plain ComfyUI does not know them;
    other face nodes and extra_model_paths.yaml may have added them already, then theirs are kept and ours come
    after."""
    try:
        import folder_paths
    except ImportError:                             # (not inside ComfyUI: nothing to register)
        return
    for kind in ("insightface", "facerestore_models"):
        try:
            folder_paths.add_model_folder_path(kind, os.path.join(folder_paths.models_dir, kind))
            paths, exts = folder_paths.folder_names_and_paths[kind]
            if exts:                                # (no extensions listed: every file counts already)
                folder_paths.folder_names_and_paths[kind] = (paths, set(exts) | {".onnx"})
        except Exception as e:  # noqa: BLE001  (an older ComfyUI: the face swap reports the missing folder itself)
            log.warning("the %s model folder could not be registered: %s", kind, e)


_face_folders()


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


class VideoChainFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        swappers = _model_files("insightface", "hyperswap") or ["hyperswap_1b_256.onnx"]
        enhancers = ["none"] + (_model_files("facerestore_models", "gpen") or [])
        return {"required": {"images": ("IMAGE",), "person": ("IMAGE",),      # (the first picture of each person batch is used)
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
        fs = FaceSwapper(det, rec, _full("insightface", swapper), enh)   # (its models leave the GPU with the job)

        def u8(t):
            return (t[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        try:
            pics = [u8(p)[0] for p in (person, person_2, person_3, person_4) if p is not None]
            identity = fs.identity(pics)
            result = images[..., :3].to("cpu", copy=True)             # one copy, filled frame by frame
            bar = comfy.utils.ProgressBar(images.shape[0])

            def frames():
                for k in range(images.shape[0]):
                    yield u8(images[k:k + 1])[0]

            def put(k, f):
                result[k] = torch.from_numpy(f).float() / 255.0
            total = images.shape[0]
            _none, done = fs.run(frames(), identity, enhance if enh else 0.0,
                                 progress=lambda k, n: bar.update_absolute(k, n), out=put, total=total)
        finally:
            del fs
        if done == 0:
            raise ValueError("no face was found to swap in this clip")
        return (result, done)


from .scale import VideoChainLanczosScale  # noqa: E402
from .h3_lora import VideoChainH3LoRA  # noqa: E402
from .pin_noise import VideoChainH3PinNoise  # noqa: E402

NODE_CLASS_MAPPINGS = {"VideoChainFaceSwap": VideoChainFaceSwap,
                       "VideoChainLanczosScale": VideoChainLanczosScale,
                       "VideoChainH3LoRA": VideoChainH3LoRA,
                       "VideoChainH3PinNoise": VideoChainH3PinNoise}
if HAS_QWEN21:
    NODE_CLASS_MAPPINGS["VideoChainQwen21FunInpaint"] = VideoChainQwen21FunInpaint
else:
    log.warning("video-chain: this ComfyUI has no Qwen-Image 2.1 Fun ControlNet internals: "
                "VideoChainQwen21FunInpaint is not offered")
NODE_DISPLAY_NAME_MAPPINGS = {"VideoChainQwen21FunInpaint": "Qwen-Image 2.1 Fun Inpaint (Promptwood)",
                              "VideoChainFaceSwap": "Face Swap, HyperSwap (Promptwood)",
                              "VideoChainLanczosScale": "Lanczos Scale, all cores (Promptwood)",
                              "VideoChainH3LoRA": "H3 LoRA, also on pruned H3 (Promptwood)",
                              "VideoChainH3PinNoise": "H3 pinned-frame noise (Promptwood)"}
