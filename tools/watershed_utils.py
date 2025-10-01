import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def watershed_skimage(gray_heatmap, min_distance=1):
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

def watershed_opencv(region_score):
    fore = np.uint8(region_score > 0.75)
    back = np.uint8(region_score < 0.05)
    unknown = 1 - (fore + back)
    ret, markers = cv2.connectedComponents(fore)
    markers += 1
    markers[unknown == 1] = 0

    labels = watershed(-region_score, markers)
    # get only foreground labels
    labels = labels * (labels > 1)
    return labels

    