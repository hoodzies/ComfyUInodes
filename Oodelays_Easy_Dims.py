class ResolutionPresetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size": (["1Mp (SDXL,FLUX CTX,2509)", "2Mp (FLUX,ZIMAGE,2511)"],),
                "ratio": (["1:1", "3:2", "4:3", "16:9"],),
                "orientation": (["Landscape", "Portrait"],)
            }
        }


    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "oodelay"


    def get_resolution(self, size, ratio, orientation="Landscape"):
        mapping = {
            "1Mp (SDXL,FLUX CTX,2509)": {
                "1:1": (1024, 1024),
                "3:2": (1216, 832),
                "4:3": (1152, 896),
                "16:9": (1344, 768)
            },
            "2Mp (FLUX,ZIMAGE,2511)": {
                "1:1": (1408, 1408),
                "3:2": (1728, 1152),
                "4:3": (1664, 1216),
                "16:9": (1920, 1088)
            }
        }

        try:
            w, h = mapping[size][ratio]
        except KeyError:
            raise KeyError(f"Unknown size/ratio combination: {size} / {ratio}")

        if orientation == "Portrait":
            if w > h:
                w, h = h, w
        else:  # Landscape
            if h > w:
                w, h = h, w

        return (w, h)



NODE_CLASS_MAPPINGS = {
    "Oodelay's easy Dimensions": ResolutionPresetNode
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Oodelay's easy Dimensions": "Oodelay's easy Dimensions"
}