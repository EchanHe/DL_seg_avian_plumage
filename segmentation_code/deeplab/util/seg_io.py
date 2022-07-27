"""
## Purpose of script: The io (e.g. write to csv) of the model
##
## Author: Yichen He
## Date: 2022/07
## Email: csyichenhe@gmail.com
"""

import json
import os, sys
import numpy as np
import pandas as pd
from datetime import date

def write_pred_contours(valid_data , pred_contours , folder,file_name , file_col_name ,
 write_index = False , is_valid = True ):
    """
    Goal: write prediction contours to DATAFRAME csv and return the panda dataframe

    params:
        valid_data: panda dataframe of validation data. used for giving file name and types
        pred_contours: prediction contours shape [batch_size , mask class numbers]
        folder: folder to save
        file_name: name of the csv file. When is None, function doesn't save the csv
    """
    # Get the name and view from Valid data
    # df_file_names = valid_data.df[[file_col_name]]
    df_file_names = valid_data.df.drop(valid_data.contour_col , axis=1 , errors = 'ignore')
    df_file_names = df_file_names.reset_index(drop=True)
    result = pd.DataFrame(pred_contours, columns = valid_data.contour_col )
    if not os.path.exists(folder):
        os.makedirs(folder)
  
    # if is_valid:
    #     gt_coords = valid_data.df[valid_data.coords_cols].values
    #     pred_coord[gt_coords==-1] = -1

    # Write the polygons in if there is given patches_coord
    # Other wise assign all -1.

    result = pd.concat([df_file_names,result],axis=1)


    if file_name is not None:
        result.to_csv(folder+file_name+".csv" , index =write_index)
    return result

def network_performance_dataframe(result_dict= {},name = None,
    mean_iou = None, mean_precision = None, mean_recall = None):
    """
    Goals: write value into dictionry, the default value is None
        which the dict can be used into grid searching result.
    """
    remove_keys = ['restore_param_file', 'config_name']
    for remove_key in remove_keys:
        if remove_key in result_dict.keys():
            result_dict.pop(remove_key)

    result_dict['name'] = name

    result_dict['mean_iou'] = mean_iou
    result_dict['mean_precision'] = mean_precision
    result_dict['mean_recall'] = mean_recall

    result_dict = {str(k):str(v) for k,v in result_dict.items()}
    return result_dict


def read_masks_from_json(path):
    with open(path, 'r') as f:
        dict = json.load(f)
    return dict

def get_info_from_params_seg(params):
    outputs = {}
    keys = ["name", "category"  , "output_stride","img_aug", "nepochs" ,"batch_size", "learning_rate",
     "decay_step" , "decay_step" ,"lambda_l2"]
    for key in keys:
        assert key in params.keys() , "No this keys, check the config file"
        outputs[key] = params[key]
    return outputs


###Deprecate

def masks_to_json(masks, path ,names =None):
    """
    Write the mask to json file.

    1 layer: File name and ID
    2 layer: id of class
    3 layer: cols and rows
    4 layer: a list of col and row index

    params:
        masks [batch , height, width, n_mask]
        path: The path for saving the json file
        names: the names of each masks
    """
    print("masks_to_json deprecated")
    # result = {}
    # for i in range(masks.shape[0]):
    #     mask = masks[i,...]
    #     if names is None:
    #         result[i] = mask_to_dict(mask)
    #     else:
    #         result[names[i]] = mask_to_dict(mask)

    # f = open(path,"w")
    # f.write(json.dumps(result))
    # f.close()

def write_seg_result(iou, params , folder ,cor_pred = None):
    print("write_seg_result deprecated")

    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    # result = {}
    # result['config'] = get_info_from_params_seg(params)
    # result["miou"] = iou
    # result["correct_predict_rate"] = cor_pred

    # result_name = "protocol_result_{}_{}.json".format(str(date.today()),params["name"])
    # print("write into: ", result_name)
    # f = open(folder+result_name,"w")
    # f.write(json.dumps(result ,indent=2 ,sort_keys=True))
    # f.close()


# def masks_to_json(masks, path ,names =None):
# """
# Write the mask to json file.

# 1 layer: File name and ID
# 2 layer: id of class
# 3 layer: cols and rows
# 4 layer: a list of col and row index

# params:
#     masks [batch , height, width, n_mask]
#     path: The path for saving the json file
#     names: the names of each masks
# """

# result = {}
# for i in range(masks.shape[0]):
#     mask = masks[i,...]
#     if names is None:
#         result[i] = mask_to_dict(mask)
#     else:
#         result[names[i]] = mask_to_dict(mask)

# f = open(path,"w")
# f.write(json.dumps(result))
# f.close()

# def mask_to_dict(mask):
# back_ground_id = 0

# n_cl = mask.shape[-1]
# result ={}
# all_rows = [0] * n_cl
# all_cols = [0] * n_cl
# for c in range(n_cl):
#     if c != back_ground_id:
#         rows , cols = np.where(mask[...,c]==1)
#         rows = rows.tolist()
#         cols = cols.tolist()
#         result[c] = {'rows':rows, 'cols':cols}
# return result