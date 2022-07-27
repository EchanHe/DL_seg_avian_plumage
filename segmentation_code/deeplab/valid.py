"""
## Purpose of script: Use for validation and predictions
##
## Author: Yichen He
## Date: 2022/07
## Email: csyichenhe@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import pandas as pd

import sys
import os
import network
## Add dir to path
dirname = os.path.dirname(__file__)
# input_lib_dir= os.path.abspath(os.path.join(dirname,"../../input"))
# util_lib_dir= os.path.abspath(os.path.join(dirname,"/util"))
# sys.path.append(input_lib_dir)
# sys.path.append(util_lib_dir)
import util.data_input
from util.plumage_config import process_config
from util.visualize_lib import *
from util.seg_io import write_pred_contours
from util.seg_util import  segs_to_masks, masks_to_contours
import util.seg_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print('--Parsing Config File')
params = process_config('config_valid.cfg')




df_pred = pd.read_csv(params['input_file'])
print("Read csv data from: ",params['input_file'], "in folder:", params['img_folder'])

valid_data = data_input.plumage_data_input(df_pred,batch_size=params['batch_size'],is_train =False,
                           pre_path =params['img_folder'],state=params['data_state'],file_col = params['file_col'],
                           scale=params['scale'] ,is_aug = False,
                           heatmap_scale = params['output_stride'],
                           contour_col_override = params.setdefault('contour_col_override', None))


if 'is_valid' in params:
    is_valid = params['is_valid']
else:
    is_valid = False


model = network.Network(params,valid_data.img_width, valid_data.img_height, is_train = False)
predict = model.deeplab_v3()



restore_file = params['restore_param_file']
initialize = params['init']
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
# Generate the result from model
with tf.Session() as sess:
    if initialize:
        print ("Initializing Network")
        sess.run(init_op)
    else:
        assert os.path.exists(restore_file + ".index") , "Ckpt file is wrong, please check the config file!"
        print("Read checkpoint from: ", restore_file)
        sess.run(init_op)
        saver.restore(sess, restore_file)

    #Generate predictions from trained network
    pred_contours = np.zeros((0, 1))
    for i_df_valid in np.arange(0,valid_data.df.shape[0],valid_data.batch_size):
        x = valid_data.get_next_batch_no_random()
        feed_dict = {
            model.images: x,

            }      
        result_mini = np.argmax(sess.run(predict, feed_dict=feed_dict) ,axis =3)   
        mask_mini = segs_to_masks(result_mini)
        pred_contours_mini = masks_to_contours(mask_mini , scale = params['scale'])
        pred_contours = np.vstack((pred_contours, pred_contours_mini))  
    pred_contours = pred_contours[:valid_data.df_size,...] 

    predicted_result = write_pred_contours(valid_data , pred_contours ,
        folder = params['valid_result_dir'],
        file_name = params['result_name'], file_col_name = params['file_col'])

if set(valid_data.contour_col).issubset(df_pred.columns) and is_valid:

    valid_data = data_input.plumage_data_input(df_pred,batch_size=params['batch_size'],is_train =True,
                           pre_path =params['img_folder'],state=params['data_state'],file_col = params['file_col'],
                           scale=params['scale'] ,is_aug = False,
                           heatmap_scale = params['output_stride'],
                           contour_col_override = params.setdefault('contour_col_override', None))

    result_data = data_input.plumage_data_input(predicted_result,batch_size=params['batch_size'],is_train =True,
                           pre_path =params['img_folder'],state=params['data_state'],file_col = params['file_col'],
                           scale=params['scale'] ,is_aug = False,
                           heatmap_scale = params['output_stride'],
                           contour_col_override = params.setdefault('contour_col_override', None))

    miou_list = np.array([])
    cor_pred_list =np.array([])
    recall_list = np.array([])
    for i_df_valid in np.arange(0,valid_data.df.shape[0],valid_data.batch_size):
        _,y_valid_mini = valid_data.get_next_batch_no_random()
        _,y_result_mini = result_data.get_next_batch_no_random()

        y_valid_mini = np.argmax(y_valid_mini,axis =3) 
        y_result_mini = np.argmax(y_result_mini,axis =3) 

        acc_iou = seg_metrics.segs_eval(y_result_mini,y_valid_mini,mode="miou" , background = 0)
        acc_cor_pred = seg_metrics.segs_eval(y_result_mini,y_valid_mini,mode="precision" , background = 0)
        recall_mini = seg_metrics.segs_eval(y_result_mini,y_valid_mini,mode="recall" , background = 0)

        miou_list = np.append(miou_list,acc_iou)
        cor_pred_list = np.append(cor_pred_list , acc_cor_pred)
        recall_list = np.append(recall_list , recall_mini)

    mean_recall = np.mean(recall_list)
    mean_miou = np.mean(miou_list)
    mean_cor_pred = np.mean(cor_pred_list)

    print("IOU:{}, precision:{}, Recall:{}".format(mean_miou , mean_cor_pred , mean_recall))
