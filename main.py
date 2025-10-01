# Standard library imports
import os
import warnings
from time import time

# Third-party imports
import cv2
import numpy as np
from craft_text_detector import Craft, craft_utils
from scipy import ndimage

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

from tools.watershed_utils import watershed_skimage, watershed_opencv

warnings.filterwarnings("ignore")

def craftshed(img_path, craft_model = Craft(cuda=False), canvas_size=1280, mag_ratio=1.0, heatmap_smoothing=0.0,
              ws_opencv=False, ws_skimage=True):

    ## Time measurement
    t0 = time()
    # Load image
    img_color,img_gray = load_image(img_path)

    # Define empty global text heatmap
    H, W = img_gray.shape[:2]

    # Resize image
    resized_image, target_ratio, size_heatmap  = resize(img_gray, canvas_size=canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # Apply craft to get text scores
    text_hmap, _ = detect_text_regions(craft_model, resized_image)

    
    t1 = time()
    print(f"Total time: {t1 - t0:.3f} seconds")

    # Normalize text_hmap to [0,1] if needed
    heatmap = normalize_hmap(text_hmap)
    # Save Text Heatmap
    cvt_hmap = cvt2HeatmapImg(heatmap)
    # resize heatmap to original image size
    if cvt_hmap.shape != img_gray.shape:
        cvt_hmap = cv2.resize(cvt_hmap, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'./heatmap.png', cvt_hmap)


    # Original Post-processing (thresholding and box extraction)
    boxes, polys = craft_utils.getDetBoxes(
        heatmap, np.zeros_like(heatmap), 0.7, 0.4, 0.4, False
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # Draw boxes for original CRAFT postprocessing
    img_boxes_CRAFT = draw_bounding_boxes_original(boxes, img_color.copy())
    # Save original CRAFT result
    cv2.imwrite(f'./th_boxes.png', img_boxes_CRAFT)

    t2 = time()
    print(f"Total time: {t2 - t1:.3f} seconds")

    ## Apply watershed
    # Optional Gaussian smoothing skimage
    if heatmap_smoothing > 0:
        heatmap = ndimage.gaussian_filter(heatmap, sigma=heatmap_smoothing)

    if ws_opencv:
        # Apply CRAFT paper watershed algorithm
        labels_ws = watershed_opencv(heatmap)
        # Draw boxes for watershed
        img_boxes = draw_bounding_boxes(labels_ws, img_gray.copy(), img_color.copy(), ratio_w, ratio_h, ratio_net=2)
        # Save watershed result
        cv2.imwrite(f'./ws_cv_boxes.png', img_boxes)
        # Save watershed labels
        ## transform labels to 0-255 and randomize colors
        labels_color = np.zeros((*labels_ws.shape, 3), dtype=np.uint8)
        ws_colors = []
        for i in range(1, labels_ws.max()+1):
            ws_colors.append(np.random.randint(0, 255, size=3))
            labels_color[labels_ws == i] = ws_colors[-1]

        if labels_color.shape[:2] != img_gray.shape:
            cv2.imwrite(f'./ws_cv_blobs.png', cv2.resize(labels_color, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST))
        else:
            cv2.imwrite(f'./ws_cv_blobs.png', labels_color)

    
    if ws_skimage:
        labels_ws = watershed_skimage(heatmap)
        # Draw boxes for watershed
        img_boxes = draw_bounding_boxes(labels_ws, img_gray.copy(), img_color.copy(), ratio_w, ratio_h, ratio_net=2)
        # Save watershed result
        cv2.imwrite(f'./ws_sk_boxes.png', img_boxes)
        # Save watershed labels
        ## transform labels to 0-255 and randomize colors
        labels_color = np.zeros((*labels_ws.shape, 3), dtype=np.uint8)
        ws_colors = []
        for i in range(1, labels_ws.max()+1):
            ws_colors.append(np.random.randint(0, 255, size=3))
            labels_color[labels_ws == i] = ws_colors[-1]

        if labels_color.shape[:2] != img_gray.shape:
            cv2.imwrite(f'./ws_sk_blobs.png', cv2.resize(labels_color, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST))
        else:
            cv2.imwrite(f'./ws_sk_blobs.png', labels_color)
    

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
    parser.add_argument("--ws_cv", type=str, default="False", help="Use OpenCV watershed")
    parser.add_argument("--ws_sk", type=str, default="True", help="Use Skimage watershed")
    args = parser.parse_args()
    img_path = args.image
    print(args)
    craftshed(
        img_path,
        craft_model=Craft(cuda=args.cuda),
        canvas_size=args.canvas_size,
        mag_ratio=args.mag_ratio,
        heatmap_smoothing=args.heatmap_smoothing,
        ws_opencv=True if args.ws_cv.lower() == 'true' else False,
        ws_skimage=True if args.ws_sk.lower() == 'true' else False,
    )

