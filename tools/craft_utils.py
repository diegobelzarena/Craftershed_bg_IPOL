import cv2
import numpy as np

def load_image(image_path):
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    return img_color, img_gray

def detect_text_regions(craft_model, image):
    prediction = craft_model.detect_text(image)
    return prediction['heatmaps']['text_score_heatmap'], prediction['heatmaps']['link_score_heatmap']

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
        cv2.rectangle(img_color, (l,t), (r,b), (0, 0, 255), 1)
    return img_color

def draw_bounding_boxes_original(bboxes, img_color):
    for bbox in bboxes:
        l = int(bbox[:,0].min())
        r = int(bbox[:,0].max())
        t = int(bbox[:,1].min())
        b = int(bbox[:,1].max())
        cv2.rectangle(img_color, (l,t), (r,b), (0, 0, 255), 1)
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

def double_image_superpixel(img):
    """
    Resize image so each pixel becomes a 2x2 block of same value.
    """
    return np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)