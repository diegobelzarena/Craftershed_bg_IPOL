import numpy as np

def get_patches(img, patch_size=1280, overlap=0.5):
    h, w = img.shape[:2]
    stride = int(patch_size * (1 - overlap))
    patches = []

    y = 0
    while y < h:
        x = 0
        patch_h = min(patch_size, h - y)
        while x < w:
            patch_w = min(patch_size, w - x)
            patch = img[y:y + patch_h, x:x + patch_w]
            patches.append((patch, (x, y)))
            if patch_w < patch_size:
                break  # reached right edge
            x += stride
        if patch_h < patch_size:
            break  # reached bottom edge
        y += stride
    return patches

def merge_heatmaps(patches, positions, image_shape):
    H, W = image_shape
    merged = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.uint8)

    for patch, (x, y) in zip(patches, positions):
        h, w = patch.shape
        merged[y:y+h, x:x+w] = np.maximum(merged[y:y+h, x:x+w], patch)
        count[y:y+h, x:x+w] += 1

    return merged