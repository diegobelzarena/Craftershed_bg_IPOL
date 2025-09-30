from time import time
import cv2
import numpy as np
from craft_text_detector import Craft, craft_utils
import os
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import math

import warnings
warnings.filterwarnings("ignore")

def load_image(image_path):
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    return img_color, img_gray

def detect_text_regions(craft_model, image):
    prediction = craft_model.detect_text(image)
    return prediction['heatmaps']['text_score_heatmap'], prediction['heatmaps']['link_score_heatmap']

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

def heatmap_watershed(gray_heatmap):
    # Ensure dtype is uint8 in [0, 255]
    if gray_heatmap.dtype != np.uint8:
        gray_heatmap = (np.clip(gray_heatmap, 0, 1) * 255).astype(np.uint8)

    # Threshold the heatmap to create a binary mask
    thresh = cv2.threshold(gray_heatmap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find local maxima in the heatmap
    local_max = peak_local_max(
        gray_heatmap, 
        min_distance=1, 
        labels=thresh,
        exclude_border=False
    )

    # Create marker image for watershed
    markers_mask = np.zeros_like(gray_heatmap, dtype=bool)
    markers_mask[tuple(local_max.T)] = True
    markers = ndimage.label(markers_mask, structure=np.ones((3, 3)))[0]

    # Apply the watershed algorithm on the inverted heatmap
    labels = watershed(-gray_heatmap, markers, mask=thresh)

    return labels

def resize(img, canvas_size, interpolation, mag_ratio=1.0):
    height, width = img.shape

    # set target image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > canvas_size:
        target_size = canvas_size

    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32), dtype=np.float32)
    resized[0:target_h, 0:target_w] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def draw_bounding_boxes(labels, img_gray, img_color,ratio_w, ratio_h, ratio_net=2):
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(labels.shape, dtype="uint8")
        mask[labels == label] = 255
        cols = np.argwhere(mask.sum(0)!=0)[:,0]
        l,r = cols[0], cols[-1]
        l = int(l * ratio_w * ratio_net)
        r = int(r * ratio_w * ratio_net)
        rows = np.argwhere(mask.sum(1)!=0)[:,0]
        t,b = rows[0], rows[-1]
        t = int(t * ratio_h * ratio_net)
        b = int(b * ratio_h * ratio_net)
        cv2.rectangle(img_color, (l,t), (r,b), (0, 255, 0), 2)
    return img_color

def draw_bounding_boxes_original(bboxes, img_color):
    for bbox in bboxes:
        l = int(bbox[:,0].min())
        r = int(bbox[:,0].max())
        t = int(bbox[:,1].min())
        b = int(bbox[:,1].max())
        cv2.rectangle(img_color, (l,t), (r,b), (0, 255, 0), 2)
    return img_color

def resize_hmap(text_hmap, patch):
    # Resize text_hmap if needed to match patch size
    if text_hmap.shape != patch.shape[:2]:
        text_hmap = cv2.resize(text_hmap, (patch.shape[1], patch.shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
    return text_hmap

def normalize_hmap(text_hmap):
    if text_hmap.dtype != np.float32:
        text_hmap = text_hmap.astype(np.float32) / 255.0
    return text_hmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def craftshed(img_path, craft_model = Craft(cuda=False), canvas_size=1280, mag_ratio=1.0,
               patch_processing=True, patch_size = 1280, overlap = 0.1, patches_folder = 'craft/patches', 
               save_patches = False):

    # Run parameters string for file names
    params = f'cs{canvas_size}_mr{mag_ratio}'
    if patch_processing:
        params += f'_pp_ps{patch_size}_ov{int(overlap*100)}'

    ## Time measurement
    t0 = time()
    # Load image
    img_color,img_gray = load_image(img_path)

    # Define empty global text heatmap
    H, W = img_gray.shape[:2]

    if patch_processing:
        # Resize image
        resized_image, target_ratio, size_heatmap  = resize(img_gray, canvas_size=canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # Define empty global text heatmap
        H2, W2 = resized_image.shape[:2]
        heatmap = np.zeros((H2//2, W2//2), dtype=np.float32)

        # Extract patches
        my_patches = get_patches(resized_image, patch_size=patch_size, overlap=overlap)

        for idx, (patch, (x_offset, y_offset)) in enumerate(my_patches):
            # Apply craft to get text scores
            text_hmap, _ = detect_text_regions(craft_model, patch)

            # Normalize text_hmap to [0,1] if needed
            text_hmap = normalize_hmap(text_hmap)
            h, w = text_hmap.shape
            
            # Save intermediate results if save_patches flag is True
            if save_patches:
                os.makedirs(patches_folder, exist_ok=True)
                patch_path = os.path.join(patches_folder, f"{params}_{idx:04d}_x{x_offset}_y{y_offset}.png")
                cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                hmap_path = os.path.join(patches_folder, f"{params}_{idx:04d}_x{x_offset}_y{y_offset}_hmap.png")
                cv2.imwrite(hmap_path, cvt2HeatmapImg(text_hmap))

            # Adjust offsets due to the 2x downscaling in heatmap of CRAFT
            y_offset//=2
            x_offset//=2

            # Merge results taking the max in overlapping regions
            heatmap[y_offset:y_offset+h, x_offset:x_offset+w] = np.maximum(
                heatmap[y_offset:y_offset+h, x_offset:x_offset+w], text_hmap)
            
    else:
        # Resize image
        resized_image, target_ratio, size_heatmap  = resize(img_gray, canvas_size=canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # Apply craft to get text scores
        text_hmap, _ = detect_text_regions(craft_model, resized_image)

        # Normalize text_hmap to [0,1] if needed
        heatmap = normalize_hmap(text_hmap)

    # Original Post-processing (thresholding and box extraction)
    boxes, polys = craft_utils.getDetBoxes(
        heatmap, np.zeros_like(heatmap), 0.7, 0.4, 0.4, False
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    # Apply watershed
    labels_ws = heatmap_watershed(heatmap)

    t1 = time()
    print(f"Total time: {t1 - t0:.3f} seconds")
    
    # Draw boxes for watershed
    img_boxes = draw_bounding_boxes(labels_ws, img_gray.copy(), img_color.copy(), ratio_w, ratio_h)
    # Draw boxes for original CRAFT postprocessing
    img_boxes2 = draw_bounding_boxes_original(boxes, img_color.copy())

    # Save results
    res_path = 'results'
    os.makedirs(res_path, exist_ok=True)
    basename = os.path.basename(img_path)
    id = os.path.splitext(basename)[0]
    
    # Save watershed result
    cv2.imwrite(f'{res_path}/{id}_ws_{params}.png', img_boxes)
    # Save original CRAFT result
    cv2.imwrite(f'{res_path}/{id}_th_{params}.png', img_boxes2)
    # Save Text Heatmap
    cv2.imwrite(f'{res_path}/{id}_hmp_{params}.png', cvt2HeatmapImg(heatmap))

    # Save watershed labels
    ## transform labels to 0-255 and randomize colors
    labels_color = np.zeros((*labels_ws.shape, 3), dtype=np.uint8)
    for i in range(1, labels_ws.max()+1):
        labels_color[labels_ws == i] = np.random.randint(0, 255, size=3)
    cv2.imwrite(f'{res_path}/{id}_wslb_{params}.png', labels_color)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--cuda", action='store_true', help="Use CUDA for computation")
    parser.add_argument("--canvas_size", type=int, default=1280, help="Canvas size for resizing the image")
    parser.add_argument("--mag_ratio", type=float, default=1.0, help="Magnification factor for resizing the image")
    parser.add_argument("--patch_processing", action='store_true', help="Flag to enable patch-wise processing")
    parser.add_argument("--patch_size", type=int, default=1280, help="Patch size for processing")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap ratio between patches")
    parser.add_argument("--patches_folder", type=str, default='craft/patches', help="Folder to save patches")
    parser.add_argument("--save_patches", action='store_true', help="Flag to save patch images and heatmaps")
    args = parser.parse_args()
    img_path = args.image
    print(args)
    craftshed(
        img_path,
        craft_model=Craft(cuda=args.cuda),
        canvas_size=args.canvas_size,
        mag_ratio=args.mag_ratio,
        patch_processing=args.patch_processing,
        patch_size=args.patch_size,
        overlap=args.overlap,
        patches_folder=args.patches_folder,
        save_patches=args.save_patches
    )

