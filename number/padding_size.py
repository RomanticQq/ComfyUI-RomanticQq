class PaddingSize:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "w_dst": ("INT", {"default": 2560, "min": 0, "max": 2048, "step": 1}),
                "h_dst": ("INT", {"default": 1080, "min": 0, "max": 2048, "step": 1}),
                "w_src": ("INT", {"default": 1024, "min": 0, "max": 2048, "step": 1}),
                "h_src": ("INT", {"default": 1536, "min": 0, "max": 2048, "step": 1}),
                "x1": ("INT", {"default": 1024, "min": 0, "max": 2048, "step": 1}),
                "y1": ("INT", {"default": 1536, "min": 0, "max": 2048, "step": 1}),
                "x2": ("INT", {"default": 500, "min": 0, "max": 2048, "step": 1}),
                "y2": ("INT", {"default": 500, "min": 0, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT", "INT","INT","INT")
    RETURN_NAMES = ("padding_left","padding_top","padding_right","padding_bottom")
    FUNCTION = "test"
    CATEGORY = "RomanticQq/number"
    """
    方案一：

    """
    def test(self, w_dst, h_dst, w_src, h_src,x1, y1, x2, y2):
        padding_top = 0
        padding_bottom = 0
        ratio = w_dst/h_dst
        x_center = (x1+x2)//2
        y_center = (y1+y2)//2
        tmp_w = max(x_center, w_src-x_center)*2
        tmp_h = tmp_w/ratio
        if tmp_h >= h_src:
            tmp_w = int(tmp_h*ratio)
            padding_top = int((tmp_h-h_src)/3)
            padding_bottom = int(tmp_h-h_src-padding_top)
        else:
            tmp_h = h_src
            tmp_w = int(tmp_h*ratio)
        padding_left = int(tmp_w/2-x_center)
        padding_right = int(tmp_w-w_src-padding_left)
        print(f"padding_left: {padding_left}, padding_top: {padding_top}, padding_right: {padding_right}, padding_bottom: {padding_bottom}")

        return (padding_left, padding_top, padding_right, padding_bottom)