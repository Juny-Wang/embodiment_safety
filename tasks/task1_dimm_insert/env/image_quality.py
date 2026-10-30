"""Conservative RGB quality checks, not task-success or camera-pose validation."""
def inspect_rgb(rgb):
    import numpy as np
    a=np.asarray(rgb)
    if a.ndim!=3 or a.shape[2] not in (3,4) or min(a.shape[:2])<1:
        return {"valid":False,"reason":"invalid_shape"}
    if a.dtype!=np.uint8:
        return {"valid":False,"reason":"expected_uint8","dtype":str(a.dtype)}
    a=a[:,:,:3]
    mean=float(a.mean());bright=float(np.mean(a.max(axis=2)>8))
    near_black=mean<1.0 and bright<.01
    return {"valid":not near_black,"reason":"near_black" if near_black else "ok",
            "mean_0_255":mean,"bright_pixel_fraction":bright,
            "policy":"mean<1 AND fraction(max_RGB>8)<0.01; data-quality heuristic"}

def inspect_file(path):
    import numpy as np
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
            return inspect_rgb(np.asarray(im))
    except Exception as e:return {"valid":False,"reason":"image_decode_error","detail":repr(e)}
