"""
## Purpose of script: Use for visualize and save image of landmarks
##
## Author: Yichen He
## Date: 2022/07
## Email: csyichenhe@gmail.com
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
import os
import cv2
import re

##### Segmentation part ##########

from seg_util import segs_to_masks,extract_classes


color_list = [np.array([255,0,0]),
              np.array([0,255,0]) ,
              np.array([0,0,255]),
              np.array([255,255,0]),
              np.array([0,255,255]) ,
              np.array([255,0,255]),
              np.array([255,0,0]),
              np.array([0,255,0]) ,
              np.array([0,0,255]),
              np.array([255,255,0]),
              np.array([0,255,255]) ,
              np.array([255,0,255])
              ]
inter_color_list = [np.array([255,0,0]),
                    np.array([0,255,0]),
                    np.array([0,0,255]) ]
def show_one_masks(plt, img, pred_mask, gt_mask, fig_name = "", save_path = None, 
    show_fig_title = True , format = 'png'):
    """
    Plot the gt or pred mask on top of a image

    params:
        plt : pyplot
        img: [height, width, 3]
        gt / pred mask: [height, width]


    """
    if show_fig_title:
        plt.title(fig_name , fontsize = 20)


    alpha = 0.5
    

    intersection = np.logical_and(gt_mask , pred_mask)
    
    color_result = np.zeros((intersection.shape))
    color_result[pred_mask==True] = 1
    color_result[gt_mask==True] = 2
    color_result[intersection==True] = 3

    ori_image = img.copy()
    for i_color in range(1,4):
        for c in range(3):
            img[:, :, c] = np.where(color_result == i_color,
                              ori_image[:, :, c] * (1 - alpha) + alpha * inter_color_list[i_color-1][c] ,
                              img[:, :, c])
        
    
    # if img is not None:
    plt.imshow(img)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.axis('off')
        plt.savefig(save_path + fig_name+'.'+format ,bbox_inches='tight',format = format,pad_inches=0)
        plt.close("all")
        plt.clf()



def save_masks_on_image(images, pred_segs, save_path, fig_names=None):
    """
    Save predction on top of images
    
    params:
        images [batch, height, width, 3]
        pred_segs [batch, height, width]  
        save_path directory of saving figures
        fig_names: a list of img names that used to name the figure
    """ 
    _check_shape_NHW(images ,pred_segs)  
    # Set the figure params
    result_size = images.shape[0]
    if result_size < 10:
        batch = result_size
    else:
        batch = 10
    nrows = 2
    ncols =5
    alpha = 0.3
    
    
    cl, n_cl = extract_classes(pred_segs)
    
    for i_start in range(0,result_size,batch):
        fig  = plt.figure(figsize=(100,40))       
        for i_row in range(i_start, i_start+batch):
            plt.subplot(nrows, ncols,(i_row)%10+1)
            image = images[i_row,...].copy()
            ori_image= images[i_row,...]
            for i_cl in range(1,n_cl):
                for c in range(3):
                    image[:, :, c] = np.where(pred_segs[i_row,...] == i_cl,
                                      ori_image[:, :, c] *
                                      (1 - alpha) + alpha * color_list[i_cl-1][c] ,
                                      image[:, :, c])
            if fig_names is not None:
                plt.title(fig_names[i_row] , fontsize = 20)   
            plt.imshow(image)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path+"seg_{}.jpg".format(i_start))
        # fig.savefig("/home/yichenhe/plumage/result_visualization/segment_contour/" + "image_and_mask{}.jpg".format(i_start))
        plt.close(fig)



def save_pred_diff_on_image(images, gt_segs, pred_segs , c , save_path , fig_names=None):
    """
    Save ground truth, prediction and their intersection of a certain class on image into figure
    
    params:
        images [batch, height, width, 3]
        pred_segs [batch, height, width] 
        gt_segs [batch, height, width] 
        c, the class of the segmentation result.
        save_path directory of saving figures
        fig_names: a list of img names that used to name the figure
    """ 

    _check_shape_NHW(images ,pred_segs, gt_segs)

    # Set the figure params
    result_size = gt_segs.shape[0]
    if result_size < 10:
        batch = result_size
    else:
        batch = 10
    nrows = 2
    ncols =5
    alpha = 0.3
    
    
    
    gt_mask = gt_segs == c
    pred_mask = pred_segs == c

    
    intersection = np.logical_and(gt_mask , pred_mask)
    
    color_result = np.zeros((intersection.shape))
    color_result[pred_mask==True] = 1
    color_result[gt_mask==True] = 2
    color_result[intersection==True] = 3
    

    
    for i_start in range(0,result_size,batch):
        fig  = plt.figure(figsize=(100,40)) 
        
        for i_row in range(i_start, i_start+batch):
            plt.subplot(nrows, ncols,(i_row)%10+1)
            image = images[i_row,...].copy()
            ori_image = images[i_row,...]
            for i_color in range(1,4):
                for c in range(3):
                    image[:, :, c] = np.where(color_result[i_row,...] == i_color,
                                      ori_image[:, :, c] * (1 - alpha) + alpha * inter_color_list[i_color-1][c] ,
                                      image[:, :, c])
            # Save the title of figure        
            if fig_names is not None:
                plt.title(fig_names[i_row] , fontsize = 20)          
            plt.imshow(image)
        fig.savefig(save_path+"segdiff_{}.jpg".format(i_start))
        plt.close(fig)

def save_masks(gt_segs, pred_segs, save_path, fig_names=None):
    """
    Save segmention of ground truth, prediction and their intersection into figure.
    
    params:
        pred_segs [batch, height, width] 
        gt_segs [batch, height, width] 
        save_path directory of saving figures
        fig_names: a list of img names that used to name the figure
    """ 

    _check_shape_NHW(pred_segs, gt_segs)

    # Set the figure params
    result_size = gt_segs.shape[0]
    if result_size < 10:
        batch = result_size
    else:
        batch = 10
    nrows = 10
    ncols =3
    
    
    intersection = (gt_segs == pred_segs)
    plot_list = [pred_segs,gt_segs,intersection]
    
    for i_start in range(0,result_size,batch):
        fig  = plt.figure(figsize=(30,100)) 
        for i_row in range(i_start, i_start+batch):
            for i_plot in range(0,ncols):    
                plt.subplot(nrows, ncols, (i_row*ncols+i_plot)%30+1)
                plt.imshow(plot_list[i_plot][i_row])

            # Save the title of figure        
            if fig_names is not None:
                plt.title(fig_names[i_row] , fontsize = 20)         
        fig.savefig(save_path+"segs_{}.jpg".format(i_start))
        plt.close(fig)


##### Segmentation part ##########


#### Coords part ######

# LM_CNT = 16

def show_markups(imgs , pred_coords = None , pred_patches =None , pred_contours=None,
    gt_coords =None , gt_patches=None, gt_contours=None, pck_threshold = None, 
    save_path = None, image_name ="markup" , sub_fig_names = None , LM_CNT=16):
    """
    Plot the markups and images
    
    params:
        img: [batch, height, width, 3]
        gt / pred_coords: coordinations [batch , landmark *2 ]    
        gt / pred_patches: a list of patches coords  [batch][patches count][patch points *2 ].
        gt / pred_contours: a list of contours coords  [batch][contours count][contour points *2 ].
        save_path directory of saving figures
        fig_name: name of the fig was saved
        sub_fig_names: a list of img names that used to name the figure
    """
    result_size = imgs.shape[0]
    if result_size < 10:
        batch = result_size
    else:
        batch = 10

    ncols = batch
    # batch_size = ncols = imgs.shape[0]
    # assert batch_size <=20, "The batch size is larger than 20, bad for plotting"
    nrows = 1

    

    for i_start in range(0,result_size,batch):
        fig  = plt.figure(figsize=(ncols * 10, nrows *10))

        for i_row in range(i_start, i_start+batch):
            plt.subplot(nrows, ncols,(i_row)%(ncols*nrows)+1)

            show_one_markup(plt, img = imgs[i_row,...] ,
            pred_coord = _none_or_element(pred_coords ,i_row) , pred_patch =_none_or_element(pred_patches ,i_row),
            pred_contour = _none_or_element(pred_contours ,i_row),
            gt_coord = _none_or_element(gt_coords ,i_row), gt_patch =   _none_or_element(gt_patches ,i_row),
            gt_contour = _none_or_element(gt_contours ,i_row),pck_threshold = pck_threshold,
            fig_name = _none_or_element(sub_fig_names ,i_row) , LM_CNT=LM_CNT)

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.axis('off')
            plt.savefig(save_path+image_name+"_{}.jpg".format(i_start) ,bbox_inches='tight')
        plt.close("all")
        fig.clf()


def show_one_markup(plt, img, pred_coord , pred_patch , pred_contour,
    gt_coord  , gt_patch, gt_contour,pck_threshold, fig_name = "" , LM_CNT =15, save_path = None,
    linewidth = 3, label_names = None,
    show_patch_labels = False , show_colour_labels = False , show_fig_title = True , format = 'png'):
    """
    Plot one image and markup using show_coords and show_patches

    params:
        plt : pyplot
        img: [height, width, 3]
        gt / pred_coord: coordinations [landmark *2 ]    
        gt / pred_patch: a list of patches coords [patches count][patch points *2 ].
        gt / pred_contour: a list of contours coords [contours count][contour points *2 ].
        fig_name: The img name that used to name current figure
    """
    if show_fig_title:
        plt.title(fig_name , fontsize = 20)
    plt.imshow(img)
    # Show the predict mark and pred_patch
    show_patches(plt, pred_patch,'deepskyblue' , show_patch_labels = show_patch_labels,linewidth=linewidth , label_names = label_names)
    show_patches(plt, pred_contour,'deepskyblue' ,linewidth=linewidth )
    show_coords(plt, pred_coord, pck_threshold,'deepskyblue' , LM_CNT = LM_CNT , show_patch_labels= show_patch_labels, label_names = label_names)

    show_patches(plt, gt_patch, 'red',linewidth=linewidth, label_names = label_names)
    show_patches(plt, gt_contour , 'red', show_patch_labels = show_patch_labels,linewidth=linewidth)
    show_coords(plt, gt_coord, pck_threshold, 'red' , LM_CNT = LM_CNT, label_names = label_names)
    
    img_height, img_width = img.shape[0:2]

    if show_colour_labels:
        plt.plot([0.8,0.85],[0.92]*2, 'deepskyblue' , lw=2 , transform=plt.gca().transAxes)
        plt.text(0.8 , 0.95,
            'Prediction', 
            fontdict={'color': "white",'size':12 },
            transform=plt.gca().transAxes)

        plt.plot([0.8,0.85],[0.87]*2, 'red', lw=2, transform=plt.gca().transAxes)
        plt.text(0.8 , 0.90,
            'Ground Truth', 
            fontdict={'color': "white",'size':12 },
            transform=plt.gca().transAxes)


    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.axis('off')
        plt.savefig(save_path + fig_name+'.'+format ,bbox_inches='tight',format = format)
        plt.close("all")
        plt.clf()

patches_names = ['s1','s2','s3','s4','s5', 'crown', 'nape','mantle', 'rump', 'tail',
'throat', 'breast', 'belly', 'tail\nunderside',
  'wing\ncoverts',   'wing\nprimaries\nsecondaries',]


patches_names = ['s1','s2','s3','s4','s5', 'crown', 'nape','mantle', 'rump', 'tail',
'throat', 'breast', 'belly', 
  'wing\ncoverts',   'wing\nprimaries\nsecondaries',]

# patches_names = np.arange(1,16).astype(str)

def show_coords(plt, coord ,pck_threshold = None, color = 'deepskyblue' , LM_CNT = 16,show_patch_labels = False , label_names = None):
    """
    Plot the 2D points on figure

    params:
        plt : pyplot
        coord [landmark *2 ]. eg [x1,y1, ... , xn, yn]
        lm_cnt: The landmark count
        color: The color of the point

    """
    if coord is None:
        return 0
    lm_cnt = LM_CNT
    cols_num_per_coord = 2
    if pck_threshold is not None:
        x = coord[0]
        y = coord[1] 
        plt.plot([x,x], [y,y+pck_threshold],color = 'white' , linewidth = 2.0)
        # plt.text(x* (1.01) , y * (1.01)  , str(pck_threshold)+" pixels", 
        #                      fontsize=10 , bbox=dict(facecolor='white', alpha=0.4))
    for i_col in range(lm_cnt):
        x = coord[ i_col * cols_num_per_coord]
        y = coord[ i_col * cols_num_per_coord +1] 
        if x >= 0 and ~np.isnan(x):
            # plt.plot(x, y, 'o' , alpha=0.8 , mew = 4 , mec = color )
            plt.scatter(x, y ,s=80, c = color , alpha =0.7 )
            if show_patch_labels:
                axis_width = np.max(plt.gca().get_xlim())
                axis_height = np.max(plt.gca().get_ylim())
                plt.text(x/axis_width , 1-(y/axis_height-0.02) ,
                    label_names[i_col], 
                    fontdict={'color': color,'size':12 },
                    transform=plt.gca().transAxes)
    # Write the labels at the text. 


def show_patches(plt,patch, color = 'deepskyblue' , show_patch_labels = False, linewidth=3 , label_names = None):
    """
    Plot the patches on figure

    params:
        plt : pyplot
        patch [patches count][patch points *2 ]. eg [x1,y1, ... , xn, yn]
    """ 
    if patch is None:
        return 0

    for id_p, p_coord in enumerate(patch):
        # print(p_coord)
        if isinstance(p_coord, (list)):
            length = len(p_coord)
        else:
            length = p_coord.shape[0]
        if length>=2:
            for i in range(0,length,2):
                x = (p_coord[i%length] , p_coord[(i+2)%length])
                y = (p_coord[(i+1)%length], p_coord[(i+3)%length])
                plt.plot(x,y,color = color,lw =linewidth , alpha =0.8)
            if show_patch_labels:
                plt.text(p_coord[0] , p_coord[1]*0.95,
                    label_names[id_p], 
                    fontdict={'color': color,'size':12 })
                
                

#### Util part ######
def _none_or_element(array , id_batch):
    if array is not None:
        return array[id_batch]
    else:
        return None


def _check_shape_NHW(a,b,*args):
    cond = (a.shape[:3] == b.shape[:3])
    for count, thing in enumerate(args): 
        assert type(a) == type(thing) , "All the args should be the same class"
        cond = cond and (a.shape[:3] == thing.shape[:3])

    if not cond:
        raise ShapeErr("NHW shape is different")
    
def _check_shape_NHWC(a,b,*args):
    cond = (a.shape == b.shape)
    for count, thing in enumerate(args): 
        assert type(a) == type(thing) , "All the args should be the same class"
        cond = cond and (a.shape == thing.shape)
    if not cond:
        raise ShapeErr("NHWC shape is different")


'''
Exceptions
'''
class ShapeErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)