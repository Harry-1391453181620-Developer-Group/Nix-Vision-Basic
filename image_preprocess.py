import numpy as np;
import PIL as pil;
def preprocess_image_to_ndarray(image_path: str, target_size: tuple[int, int] = (256, 256), normalize: bool = True) -> np.ndarray:
    image = pil.Image.open(image_path).convert("L").resize(target_size)
    image_array = np.array(image, dtype=np.float64)
    if normalize:
        image_array /= 255.0
    return image_array