# Standard library imports
import os
import warnings
from time import time

# Third-party imports
import cv2
import numpy as np
from craft_text_detector import Craft, craft_utils
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Local imports
from tools.craft_utils import (
    load_image, 
    detect_text_regions, 
    resize, 
    draw_bounding_boxes,
    draw_bounding_boxes_original, 
    normalize_hmap, 
    cvt2HeatmapImg, 
)
from tools.patches_utils import get_patches

warnings.filterwarnings("ignore")


def heatmap_watershed(gray_heatmap, min_distance=1):
    # Ensure dtype is uint8 in [0, 255]
    if gray_heatmap.dtype != np.uint8:
        gray_heatmap = (np.clip(gray_heatmap, 0, 1) * 255).astype(np.uint8)

    # Threshold the heatmap to create a binary mask
    thresh = cv2.threshold(gray_heatmap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find local maxima in the heatmap
    local_max = peak_local_max(
        gray_heatmap, 
        min_distance=min_distance, 
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



def craftshed(img_path, craft_model = Craft(cuda=False), canvas_size=1280, mag_ratio=1.0, heatmap_smoothing=0.0,
               patch_processing=True, patch_size = 1280, overlap = 0.1, patches_folder = 'craft/patches', 
               save_patches = False):

    # Run parameters string for file names
    params = f'cs{canvas_size}_mr{mag_ratio}_hs{heatmap_smoothing}'
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
    # Optional Gaussian smoothing skimage
    if heatmap_smoothing > 0:
        heatmap = ndimage.gaussian_filter(heatmap, sigma=heatmap_smoothing)
    labels_ws = heatmap_watershed(heatmap)

    t1 = time()
    print(f"Total time: {t1 - t0:.3f} seconds")
    
    # Draw boxes for watershed
    img_boxes = draw_bounding_boxes(labels_ws, img_gray.copy(), img_color.copy(), ratio_w, ratio_h, ratio_net=2)
    # Draw boxes for original CRAFT postprocessing
    img_boxes2 = draw_bounding_boxes_original(boxes, img_color.copy())

    # Save results
    res_path = 'results'
    os.makedirs(res_path, exist_ok=True)
    basename = os.path.basename(img_path)
    id = os.path.splitext(basename)[0]
    
    # Save watershed result
    cv2.imwrite(f'./ws_boxes.png', img_boxes)
    # Save original CRAFT result
    cv2.imwrite(f'./th_boxes.png', img_boxes2)
    # Save Text Heatmap
    cvt_hmap = cvt2HeatmapImg(heatmap)
    # resize heatmap to original image size
    if cvt_hmap.shape != img_gray.shape:
        cvt_hmap = cv2.resize(cvt_hmap, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'./heatmap.png', cvt_hmap)

    t2 = time()
    print(f"Total time: {t2 - t1:.3f} seconds")

    # Save watershed labels
    ## transform labels to 0-255 and randomize colors
    labels_color = np.zeros((*labels_ws.shape, 3), dtype=np.uint8)
    ws_colors = []
    for i in range(1, labels_ws.max()+1):
        ws_colors.append(np.random.randint(0, 255, size=3))
        labels_color[labels_ws == i] = ws_colors[-1]

    if labels_color.shape[:2] != img_gray.shape:
        cv2.imwrite(f'./ws_blobs.png', cv2.resize(labels_color, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_AREA))
    else:
        cv2.imwrite(f'./ws_blobs.png', labels_color)

    t3 = time()
    print(f"Total time: {t3 - t2:.3f} seconds")


    

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--cuda", action='store_true', help="Use CUDA for computation")
    parser.add_argument("--canvas_size", type=int, default=1280, help="Canvas size for resizing the image")
    parser.add_argument("--mag_ratio", type=float, default=1.0, help="Magnification factor for resizing the image")
    parser.add_argument("--heatmap_smoothing", type=float, default=0.0, help="Gaussian smoothing sigma for heatmap")
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
        heatmap_smoothing=args.heatmap_smoothing,
        patch_processing=args.patch_processing,
        patch_size=args.patch_size,
        overlap=args.overlap,
        patches_folder=args.patches_folder,
        save_patches=args.save_patches
    )

