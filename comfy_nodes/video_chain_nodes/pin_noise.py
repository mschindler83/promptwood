"""VideoChainH3PinNoise: the noise level MiniMax H3 conditions its pinned frames (start frame, carry-over, keyframes)
and reference pictures at. ComfyUI's H3 reads it from the conditioning ("minimax_visual_cond_noise_aug"; 0.999,
nearly clean, when it is not set) and gives those rows that timestep: a lower level mixes a little noise into them
and tells the model so, which keeps it from copying their fine detail exactly (its own artifacts too). No core node
sets it."""

KEY = "minimax_visual_cond_noise_aug"


class VideoChainH3PinNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",),
                             "visual": ("FLOAT", {"default": 0.999, "min": 0.5, "max": 1.0, "step": 0.001})}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"

    def apply(self, conditioning, visual):
        return ([[t, {**d, KEY: float(visual)}] for t, d in conditioning],)


NODE_CLASS_MAPPINGS = {"VideoChainH3PinNoise": VideoChainH3PinNoise}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoChainH3PinNoise": "H3 pinned-frame noise (Promptwood)"}
