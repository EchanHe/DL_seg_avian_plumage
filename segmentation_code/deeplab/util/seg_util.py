"""
## Purpose of script: The transformation and data processing of segs and masks
##
## Author: Yichen He
## Date: 2022/07
## Email: csyichenhe@gmail.com
"""

import numpy as np
from skimage import morphology,measure
import cv2

def masks_to_contours(masks , scale , ignore_mask_channel = [0]):
    masks_list = []
    for mask in masks:
        masks_list+= [mask_to_contour(mask , scale , ignore_mask_channel)]

    return np.array(masks_list)
def mask_to_contour(mask , scale , ignore_mask_channel):
    channel_list = []
    for channel in range(0,mask.shape[-1]):
        if channel not in ignore_mask_channel:
            contours, hierarchy = cv2.findContours(mask[...,channel].astype("uint8"),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours_list = []
            if len(contours)>0:
                for cons in contours:
                    contour_list=[]
                    for pts in cons:
                #         print(pts)
                        contour_list.append(pts[0][0]*scale)
                        contour_list.append(pts[0][1]*scale)
                    contours_list.append(contour_list)
                
                
                #contours = measure.find_contours(mask[...,channel],0.5)
                # contours_list = []
                # if len(contours)==0:
                #     print(0)
                #     print(np.sum(mask[...,channel]))
                # for contour in contours:
                #     contour = (np.flip(contour,axis = 1)*scale).astype(int)
                #     contour = contour.ravel().tolist()
                #     contours_list+=[contour]

                contours_list = str(contours_list)[1:-1]
            else:
                contours_list = np.nan
            channel_list += [contours_list]    
    return channel_list
def resize_segs(segs ,  width, height):
    """
    Resize the segmentations to given width and height

    params:
        segs [batch, height, width]
        width: width of the segmentations
        height: height of the segmentations
    

    """
    masks = segs_to_masks(segs)
    new_mask = np.zeros((masks.shape[0] , height, width, masks.shape[-1]))
    for i in range(masks.shape[0]):
        new_mask[i,...] = cv2.resize(masks[i,...] , (width, height))
    new_segs = np.argmax(new_mask, 3)
    new_mask = segs_to_masks(new_segs)
    
    return new_mask


def to_convex_hull(pred_segs):
    """
    Convert the pred segmentations to a convex hull

    params:
        pred_segs [batch, height, width]
    """
    pred_masks = segs_to_masks(pred_segs)
    n_cl = pred_masks.shape[-1]
    n_rows = pred_masks.shape[0]
    
    masks_convex_hull = np.zeros(pred_masks.shape)
    for row in range(0, n_rows):
        for cl in range(1, n_cl):
            masks_convex_hull[row,...,cl] = morphology.convex_hull_image(pred_masks[row,...,cl])
            
    
    return np.argmax(masks_convex_hull , 3)

def segs_to_masks(segs,n_cl=None):
    assert len(segs.shape) ==3 , "Make sure input is [batch_size , height, width]"
    df_size = segs.shape[0]
    # if n_cl is None:
    #     cl, n_cl = extract_classes(segs[0,...] )
    # else:
    #     cl, _ = extract_classes(segs[0,...] )
    if n_cl is None:
        n_cl = np.max(segs) + 1

    masks = np.zeros((segs.shape[0] , segs.shape[1] , segs.shape[2] , n_cl))
    for i in np.arange(df_size):
        cl, _ = extract_classes(segs[i,...] )
        # print(np.sum(extract_masks(segs[i,...] , cl, n_cl)))
        masks[i,...] = extract_masks(segs[i,...] , cl, n_cl)

    return masks

#seg [height , width]
def seg_to_mask(seg,n_cl=None):

    assert len(seg.shape) ==2 , "Make sure input is [height, width]"

    if n_cl is None:
        n_cl = np.max(seg) + 1
    cl, _ = extract_classes(seg)
    return extract_masks(seg , cl, n_cl)


'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]



def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((h, w , n_cl))
    # print(masks.shape)
    for i, c in enumerate(cl):
        # print(c)
        masks[:, : , i] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def segs_size(segs):
    try:
        df_size = segs.shape[0]
        height = segs.shape[1]
        width  = segs.shape[2]
    except IndexError:
        raise

    return df_size , height, width

def check_df_size(pred_segms, gt_segms):
    p_a,p_b,p_c = segs_size(pred_segms)
    g_a, g_b, g_c = segs_size(gt_segms)
    if (p_a != g_a) or (p_b != g_b) or (p_c!=g_c):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")
'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

# Deprecated

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    print("Deprecated")
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask