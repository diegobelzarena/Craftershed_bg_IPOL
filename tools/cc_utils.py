import numpy as np
from .craft_utils import double_image_superpixel


def ws_cc_assignment(labels_ws, labels_ch):

    # If ws labels are double the size of ch labels, resize them
    if (labels_ws.shape[0]*2) == labels_ch.shape[0]:
        print("Resizing ws labels to match ch labels")
        labels_ws = double_image_superpixel(labels_ws)
    if labels_ws.shape != labels_ch.shape:
        raise ValueError("labels_ws and labels_ch must have the same shape")
    
    # zero array for counting character components in each box
    lb_by_lb = np.zeros((labels_ws.max()+1, labels_ch.max()+1))
    # zero array for mapping ws labels to ch labels
    ws_map = np.zeros(labels_ch.shape)
    for label in np.unique(labels_ws)[:]:
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(labels_ws.shape, dtype="uint8")
        mask[labels_ws == label] = 255
        cols = np.argwhere(mask.sum(0)!=0)[:,0]
        l,r = cols[0], cols[-1]
        rows = np.argwhere(mask.sum(1)!=0)[:,0]
        t,b = rows[0], rows[-1]
        # get center of bounding box in ws
        #ws_mean = np.array([((l+r)/2), ((t+b)/2)])
        ws_map[t:b, l:r] = label
        for k in np.unique(labels_ch[t:b, l:r]):
            if k == 0:
                continue
            # get barycenter of mask in ch
            label_mask = np.zeros(labels_ch.shape, dtype="uint8")
            label_mask[labels_ch == k] = 1
            # save distance between means
            lb_by_lb[label, k] = np.sum(np.sum(label_mask[t:b,l:r]*(mask[t:b, l:r]!=0)))

    return lb_by_lb

def assign_ws_cc(lb_by_lb, labels_cc, ws_colors):
    assigns = np.argmax(lb_by_lb, axis=0)
    nbboxes = []
    mask = np.zeros((*labels_cc.shape,3), dtype=np.uint8)
    for k in np.unique(assigns)[:]:
        if k == 0:
            continue
        ccs = np.argwhere(assigns == k)
        for cc in ccs:
            mask[labels_cc == cc] = ws_colors[k-1]
        cols = np.argwhere(mask.sum(0)!=0)[:,0]
        l,r = cols[0], cols[-1]
        rows = np.argwhere(mask.sum(1)!=0)[:,0]
        t,b = rows[0], rows[-1]
        nbboxes.append([t,b,l,r])

    return mask, nbboxes