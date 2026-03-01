import torch
import numpy as np
from PIL import Image, ImageFilter

class CombinePresetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background": ("IMAGE",),
                "overlay": ("IMAGE",),
                "horizontal_pixel_displacement": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1.0}),
                "vertical_pixel_displacement": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1.0}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 9999.0, "step": 0.001}),
                "overlay_brightness": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "overlay_contrast": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "overlay_gamma": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "overlay_temperature": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "alpha_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "alpha_contrast": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "combine_background_overlay"
    CATEGORY = "oodelay"

    def combine_background_overlay(self, background, overlay, horizontal_pixel_displacement, vertical_pixel_displacement, scale, overlay_brightness=0.0, overlay_contrast=0.0, overlay_gamma=0.0, overlay_temperature=0.0, alpha_blur=0.0, alpha_contrast=0.0, mask=None):
        results = []
        output_masks = []

        # Background (first image only)
        bg = background[0].cpu().numpy()
        bg = np.clip(bg * 255, 0, 255).astype(np.uint8)

        if bg.shape[2] == 4:
            bg_img = Image.fromarray(bg, "RGBA")
            bg_has_alpha = True
        else:
            bg_img = Image.fromarray(bg, "RGB")
            bg_has_alpha = False

        bg_w, bg_h = bg_img.size
        bg_center_x = bg_w // 2
        bg_center_y = bg_h // 2

        for i in range(overlay.shape[0]):
            ov = overlay[i].cpu().numpy()
            ov = np.clip(ov * 255, 0, 255).astype(np.uint8)

            if ov.shape[2] == 4:
                ov_img = Image.fromarray(ov, "RGBA")
            else:
                ov_img = Image.fromarray(ov, "RGB")

            # (scaling moved later to after filters for better quality)

            # ---- OVERLAY BRIGHTNESS / CONTRAST (affect only overlay before compositing) ----
            try:
                ob = float(overlay_brightness)
                oc = float(overlay_contrast)
            except Exception:
                ob = 0.0
                oc = 0.0

            # prepare array for possible modifications (always create so gamma/temperature can run)
            tmp = ov_img.convert("RGBA")
            arr = np.array(tmp).astype(np.float32)
            rgb = arr[:, :, :3]
            a_ch = arr[:, :, 3]

            # map contrast param so 0.0 = no change
            factor = 1.0 + oc

            modified = False

            if (abs(ob) > 1e-6) or (abs(oc) > 1e-6):
                # contrast around mid-point 127.5, brightness additive in 0-255 space
                rgb = (rgb - 127.5) * factor + 127.5 + (ob * 255.0)
                rgb = np.clip(rgb, 0.0, 255.0)

                arr[:, :, :3] = rgb
                arr[:, :, 3] = a_ch
                modified = True

            # ---- OVERLAY GAMMA (decimal, 0.0 = no change) applied to overlay RGB ----
            try:
                og = float(overlay_gamma)
            except Exception:
                og = 0.0

            if abs(og) > 1e-6:
                try:
                    if og >= 0.0:
                        g = 1.0 / (1.0 + og)
                    else:
                        g = 1.0 - og
                    if g <= 0.0:
                        g = 1e-6
                    rgb_norm = np.clip(arr[:, :, :3] / 255.0, 0.0, 1.0)
                    rgb_norm = np.power(rgb_norm, g)
                    arr[:, :, :3] = (rgb_norm * 255.0)
                    modified = True
                except Exception:
                    pass

            # ---- OVERLAY TEMPERATURE (warm/cold) ----
            try:
                ot = float(overlay_temperature)
            except Exception:
                ot = 0.0

            if abs(ot) > 1e-6:
                try:
                    rgbf = arr[:, :, :3].astype(np.float32)
                    if ot > 0:
                        # warm: boost R, slight boost G, reduce B
                        rgbf[:, :, 0] = np.clip(rgbf[:, :, 0] + ot * 50.0, 0.0, 255.0)
                        rgbf[:, :, 1] = np.clip(rgbf[:, :, 1] + ot * 10.0, 0.0, 255.0)
                        rgbf[:, :, 2] = np.clip(rgbf[:, :, 2] - ot * 30.0, 0.0, 255.0)
                    else:
                        tt = -ot
                        # cold: boost B, slight boost G, reduce R
                        rgbf[:, :, 0] = np.clip(rgbf[:, :, 0] - tt * 30.0, 0.0, 255.0)
                        rgbf[:, :, 1] = np.clip(rgbf[:, :, 1] + tt * 10.0, 0.0, 255.0)
                        rgbf[:, :, 2] = np.clip(rgbf[:, :, 2] + tt * 50.0, 0.0, 255.0)
                    arr[:, :, :3] = rgbf
                    modified = True
                except Exception:
                    pass

            if modified:
                ov_img = Image.fromarray(arr.astype(np.uint8), "RGBA")
                # keep as RGBA; downstream code will handle alpha

            # ---- MASK (INVERTED LOGIC) ----
            paste_mask = None

            if mask is not None:
                mask_idx = min(i, mask.shape[0] - 1)
                m = mask[mask_idx].cpu().numpy()
                m = np.clip(m * 255, 0, 255).astype(np.uint8)
                mask_img = Image.fromarray(m, "L")

                if mask_img.size != ov_img.size:
                    mask_img = mask_img.resize(ov_img.size, Image.LANCZOS)

                inverted_mask = Image.eval(mask_img, lambda x: 255 - x)

                if ov_img.mode == "RGBA":
                    ov_alpha = np.array(ov_img.split()[3], dtype=np.float32) / 255.0
                    inv_alpha = np.array(inverted_mask, dtype=np.float32) / 255.0
                    combined_alpha = (ov_alpha * inv_alpha * 255).astype(np.uint8)
                    ov_img.putalpha(Image.fromarray(combined_alpha, "L"))
                    paste_mask = Image.fromarray(combined_alpha, "L")
                else:
                    ov_img.putalpha(inverted_mask)
                    paste_mask = inverted_mask
            else:
                if ov_img.mode == "RGB":
                    ov_img.putalpha(Image.new("L", ov_img.size, 255))
                paste_mask = ov_img.split()[3]

            # ---- ALPHA BLUR (Gaussian) applied only to the overlay alpha mask ----
            try:
                ab = float(alpha_blur)
            except Exception:
                ab = 0.0

            if paste_mask is not None and ab > 0.0:
                try:
                    paste_mask = paste_mask.filter(ImageFilter.GaussianBlur(radius=ab))
                except Exception:
                    pass

            

            # ---- SCALE OVERLAY (after filters for better quality) ----
            try:
                sc = float(scale)
            except Exception:
                sc = 1.0

            if sc != 1.0:
                new_w = max(1, int(ov_img.width * sc))
                new_h = max(1, int(ov_img.height * sc))
                ov_img = ov_img.resize((new_w, new_h), Image.LANCZOS)
                if paste_mask is not None:
                    try:
                        paste_mask = paste_mask.resize((new_w, new_h), Image.LANCZOS)
                    except Exception:
                        paste_mask = paste_mask.resize((new_w, new_h))

            # ---- PIXEL POSITION (CENTER-BASED) ----
            x = bg_center_x - (ov_img.width // 2) + horizontal_pixel_displacement
            y = bg_center_y - (ov_img.height // 2) - vertical_pixel_displacement

            # ---- PREPARE RESULT IMAGE ----
            if bg_has_alpha:
                result = bg_img.copy()
                output_mask_img = bg_img.split()[3].copy()
            else:
                result = Image.new("RGBA", bg_img.size, (0, 0, 0, 0))
                result.paste(bg_img, (0, 0))
                output_mask_img = Image.new("L", bg_img.size, 255)

            # ---- PASTE OVERLAY ----
            result.paste(ov_img, (x, y), paste_mask)

            # ---- UPDATE OUTPUT MASK ----
            temp_mask = Image.new("L", result.size, 0)
            temp_mask.paste(paste_mask, (x, y))

            out_arr = np.array(output_mask_img, dtype=np.float32)
            tmp_arr = np.array(temp_mask, dtype=np.float32)
            output_mask_img = Image.fromarray(
                np.maximum(out_arr, tmp_arr).astype(np.uint8),
                "L"
            )

            # ---- CONVERT BACK TO TENSOR ----
            result_np = np.array(result)

            # ---- ALPHA CONTRAST ADJUST (preserve pure 0 and 1) ----
            if result_np.shape[2] == 4:
                a = result_np[:, :, 3].astype(np.float32) / 255.0
                c = float(alpha_contrast)
                # clamp sensible range for safety
                if c <= -0.99:
                    c = -0.99
                # compute gamma: c==0 -> gamma==1 (no change)
                if c >= 0.0:
                    gamma = 1.0 / (1.0 + c)
                else:
                    gamma = 1.0 - c

                a = np.clip(a, 0.0, 1.0)
                a = np.power(a, gamma)
                result_np[:, :, 3] = (a * 255.0).astype(np.uint8)

            if not bg_has_alpha:
                alpha = result_np[:, :, 3:4] / 255.0
                rgb = result_np[:, :, :3]
                white = np.ones_like(rgb) * 255
                result_np = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

            result_tensor = torch.from_numpy(result_np).float() / 255.0
            mask_tensor = torch.from_numpy(np.array(output_mask_img)).float() / 255.0

            results.append(result_tensor)
            output_masks.append(mask_tensor)

        return (torch.stack(results), torch.stack(output_masks))

NODE_CLASS_MAPPINGS = {
    "Oodelay's Magical Overlay Combiner": CombinePresetNode
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Oodelay's Magical Overlay Combiner": "Oodelay's Magical Overlay Combiner"

}