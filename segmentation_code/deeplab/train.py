"""
## Purpose of script: Training the model
##
## Author: Yichen He
## Date: 2022/07
## Email: csyichenhe@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf

import numpy as np
import pandas as pd

import sys
import os
import network
import itertools
import datetime

from sklearn.model_selection import train_test_split, KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import util.data_input
from util.plumage_config import process_config, generate_grid_params, extract_config_name , config_training_steps
from util.seg_io import  *
import util.seg_metrics
from util.seg_util import masks_to_contours , segs_to_masks


def read_csv(params):
    if 'input_file' in params:
        ##reading data
        df_all = pd.read_csv(params['input_file'])
        
        # Split training and valdiation data, via category.
        if 'category_col' in params and params['category_col'] is not None:
            category_col = params['category_col']
        else:
            category_col = "view"

        if category_col not in df_all.columns:
            df_all[category_col] = 1

        gb = df_all.groupby("view")
        if params['kfold'] ==None:
            print("train_test_split with seed:",params['split_seed'] )
            # train_test_split split option
            split_list = [t for x in gb.groups for t in train_test_split(gb.get_group(x),
             test_size=params['test_size'], random_state =params['split_seed'])]
            
            df_train = pd.concat(split_list[0::2],sort = False)
            df_valid = pd.concat(split_list[1::2],sort = False)
        else:
            # Kfold option
            print("Kfold with seed: {} and {} th fold".format(params['split_seed'] ,params['kfold'] ))
            kf = KFold(n_splits=5 ,shuffle = True, random_state=params['split_seed'])
            train_list = []
            valid_list = []
            for key in gb.groups:
                data_view = gb.get_group(key)
                for idx, (train_index, valid_index) in enumerate(kf.split(data_view)):
                    if idx ==params['kfold']:
                        train_list.append(data_view.iloc[train_index,:])         
                        valid_list.append(data_view.iloc[valid_index,:]) 

            df_train = pd.concat(train_list,sort = False)
            df_valid = pd.concat(valid_list,sort = False)

    else:
        df_train = pd.read_csv(params['train_file'])
        df_valid = pd.read_csv(params['valid_file'])
    #Sampling a sub set
    if 'small_data' in params and params['small_data']:
        df_train = df_train.sample(n=10,random_state=3)
        df_valid = df_valid.sample(n=5,random_state=3)


    if 'aug_folders' in params and params['aug_folders'] is not None:
        if 'aug_file' in params and params['aug_file'] is not None: 
            aug_folder = params['aug_folders']
            df_aug = pd.read_csv(params['aug_file'] , index_col = 'file.vis')
            df_aug = df_aug.loc[df_train['file.vis'],:]
            df_aug.reset_index(inplace = True)

            df_aug['file.vis'] = aug_folder + df_aug['file.vis']

            df_train = pd.concat([df_train,df_aug]).reset_index(drop = True )


    # Create the name using some of the configuratation.
    print(params['category'])
    if params['category'] is not None and params['category'] !='all':
        # params['network_name'] +='_' + params['category']
        df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
        df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
    # else:
        # params['network_name'] +='_' + 'all'
    return df_train, df_valid

def create_data(params, df_train, df_valid):

    ### Read the training data and validation data ###
    print("Read training data ....")
    train_data = data_input.plumage_data_input(df_train,batch_size=params['batch_size'],is_train =params['is_train'],
                                pre_path =params['img_folder'],state=params['data_state'],file_col = params['file_col'],
                                scale=params['scale'] ,
                                contour_col_override = params.setdefault('contour_col_override', None))
    print("Read valid data ....\n")
    valid_data = data_input.plumage_data_input(df_valid,batch_size=params['batch_size'],is_train =params['is_train'],
                                pre_path =params['img_folder'],state=params['data_state'], file_col = params['file_col'],
                                scale=params['scale'],
                                contour_col_override = params.setdefault('contour_col_override', None))
    extract_config_name(params)
    return train_data, valid_data


def trainining(params, train_data, valid_data):

    config_name = params['config_name']
    config_training_steps(params , train_data.df_size)

    tf.reset_default_graph()
    params['input_channel'] = train_data.input_channel
    model = network.Network(params,train_data.img_width, train_data.img_height)
    predict = model.deeplab_v3()
    loss = model.loss()
    train_op = model.train_op(loss, model.global_step)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        if not os.path.exists(params['saver_directory']):
            os.makedirs(params['saver_directory'])
        if os.listdir(params['saver_directory']) == [] or params['init']:
            print ("Initializing Network")
            sess.run(init_op)
        else:
            print("Restore file from: {}".format(params['restore_param_file']))
            sess.run(init_op)
            saver.restore(sess, params['restore_param_file'])

        #### Get the summary of training and weight.
        train_summary = tf.summary.merge_all('train')
        # weight_summary = tf.summary.merge_all('weight')
        writer = tf.summary.FileWriter(os.path.join(params['log_dir'], config_name), sess.graph)

            
        for i in range(params["total_steps"]):
            ####### Training part ########
            # Get input data and label from Training set, randomly.
            tmp_global_step = model.global_step.eval()
            x_train,y_train = train_data.get_next_batch()
            # print(np.count_nonzero(vis_mini==0))
            feed_dict = {
                        model.images: x_train,
                        model.labels: np.argmax(y_train,axis = 3),
                        model.labels_by_classes: y_train
                        }
            sess.run(train_op, feed_dict=feed_dict)

            ###### Train Summary part #####
            if (i+1) % params["summary_steps"]== 0 or i == 0:
                print("{} steps Loss: {}".format(i+1,sess.run(loss, feed_dict=feed_dict)))
                lear = model.learning_rate.eval()
                # print("\tGlobal steps and learning rates: {}  {}".format(tmp_global_step,lear))
                temp_summary = sess.run(train_summary, feed_dict=feed_dict)    
                writer.add_summary(temp_summary, tmp_global_step)
                
                # For the eye detection part
                # result_mini = np.argmax(sess.run(predict, feed_dict=feed_dict) ,axis =3)   
                # print(seg_metrics.segs_eval(result_mini,np.argmax(y_train,axis = 3),mode="miou" , background = 0))
                # print(seg_metrics.segs_eval(result_mini,np.argmax(y_train,axis = 3),mode="precision" , background = 0))
                # print(np.sum(result_mini==1))
                # print(np.sum(np.argmax(y_train,axis = 3)==1))
            ######Validating the result part#####    
            if (i+1) % params["valid_steps"] ==0 or i == 0:
                #Validation part
                #write the validation result
                miou_list = np.array([])
                cor_pred_list =np.array([])
                loss_list = np.array([])
                for i_df_valid in np.arange(0,valid_data.df.shape[0],valid_data.batch_size):
                    x,y_valid = valid_data.get_next_batch_no_random()
                    y_valid_segs = np.argmax(y_valid,axis = 3)
                    feed_dict = {
                        model.images: x,
                        model.labels: y_valid_segs,
                        model.labels_by_classes: y_valid
                        }      
                    _loss = sess.run(loss, feed_dict=feed_dict)
                    result_mini = np.argmax(sess.run(predict, feed_dict=feed_dict) ,axis =3)       

                    acc_iou = seg_metrics.segs_eval(result_mini,y_valid_segs,mode="miou" , background = 0)
                    acc_cor_pred = seg_metrics.segs_eval(result_mini,y_valid_segs,mode="precision" , background = 0)
                    recall_mini = seg_metrics.segs_eval(result_mini,y_valid_segs,mode="recall" , background = 0)


                    miou_list = np.append(miou_list,acc_iou)
                    loss_list = np.append(loss_list,_loss)
                    cor_pred_list = np.append(cor_pred_list , acc_cor_pred)

                mean_loss = np.mean(loss_list)
                mean_miou = np.mean(miou_list)
                mean_cor_pred = np.mean(cor_pred_list)

                summary = sess.run(model.valid_summary,
                    feed_dict = { model.miou:mean_miou, 
                                    model.valid_loss:mean_loss,
                                    model.precision:mean_cor_pred})
                writer.add_summary(summary , tmp_global_step)  
            ####### Save the parameters to computers.
            if (i + 1) % params["saver_steps"] == 0:        
                tmp_global_step = model.global_step.eval()
                epochs = (tmp_global_step*params["batch_size"])//params["training_set_size"]
                model.save(sess, saver, params['config_name'], epochs)  

        params['restore_param_file'] = "{}-{}".format(params['config_name'], epochs)
    return model, predict    

def get_and_eval_result(params, valid_data):
    params_valid = params.copy()
    params_valid['is_train'] = False
    params_valid['l2'] = 0.0

    tf.reset_default_graph()
    model = network.Network(params_valid,valid_data.img_width, valid_data.img_height)
    predict = model.deeplab_v3()

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    # Get the predictions:
    with tf.Session() as sess:
        sess.run(init_op)
        try:
            saver.restore(sess, params_valid['saver_directory'] + params_valid["restore_param_file"])
        except:
            saver.restore(sess,params_valid["restore_param_file"])

        pred_contours = np.zeros((0, 1))

        miou_list = np.array([])
        cor_pred_list =np.array([])
        recall_list = np.array([])
        for i_df_valid in np.arange(0,valid_data.df.shape[0],valid_data.batch_size):
            x,y_valid = valid_data.get_next_batch_no_random()
            y_valid_segs = np.argmax(y_valid,axis = 3)
            feed_dict = {
                model.images: x,
                model.labels: y_valid_segs,
                model.labels_by_classes: y_valid
                }      
            result_mini = np.argmax(sess.run(predict, feed_dict=feed_dict) ,axis =3)   
            mask_mini = segs_to_masks(result_mini)
            #print(mask_mini.shape)
            pred_contours_mini = masks_to_contours(mask_mini , scale = params_valid['scale'])
            #print(pred_contours_mini.shape)
            pred_contours = np.vstack((pred_contours, pred_contours_mini))  
  
            acc_iou = seg_metrics.segs_eval(result_mini,y_valid_segs,mode="miou" , background = 0)
            acc_cor_pred = seg_metrics.segs_eval(result_mini,y_valid_segs,mode="precision" , background = 0)
            recall_mini = seg_metrics.segs_eval(result_mini,y_valid_segs,mode="recall" , background = 0)

            miou_list = np.append(miou_list,acc_iou)
            cor_pred_list = np.append(cor_pred_list , acc_cor_pred)
            recall_list = np.append(recall_list , recall_mini)
        mean_recall = np.mean(recall_list)
        mean_miou = np.mean(miou_list)
        mean_cor_pred = np.mean(cor_pred_list)

        pred_contours = pred_contours[:valid_data.df_size,...] 
    print(mean_miou , mean_cor_pred , mean_recall)
    ## Create patches and calculate the pixels inside the patch and correlation.

    write_pred_contours(valid_data , pred_contours ,
        folder = params_valid['valid_result_dir']+"grid_temp/",
        file_name = params['config_name'], file_col_name = params['file_col'])

    result_dict = network_performance_dataframe(result_dict = params_valid,
        mean_iou = mean_miou, mean_precision = mean_cor_pred, mean_recall = mean_recall)

    result_dict['result_names'] = params['config_name'] + ".csv"

    return result_dict

# read the config file part
args = sys.argv
if len(args)==2:
    config_name = args[1]
else:
    config_name = 'config_contour.cfg'

##

params = process_config(os.path.join(dirname, config_name))
grid_params = generate_grid_params(params)
print(grid_params)
# Trainning and validation
################
if bool(grid_params):

    keys, values = zip(*grid_params.items())
    final_grid_df = pd.DataFrame()
    
    #Generate parameters for grid search    
    for id_grid,v_pert in enumerate(itertools.product(*values)):
        config_name = ""
        for key, value in zip(keys, v_pert):
            params[key] = value
            config_name += "{}-{};".format(key,value)
        df_train,df_valid = read_csv(params)
        train_data, valid_data = create_data(params, df_train,df_valid)    

        ##### Create the network using the hyperparameters. #####
        model , predict = trainining(params, train_data, valid_data)

        # produce the result;
        result_dict = get_and_eval_result(params , valid_data)

        final_grid_df = final_grid_df.append(pd.DataFrame(result_dict, index=[id_grid]))

    currentDT = datetime.datetime.now()
    currentDT_str = currentDT.strftime("%Y-%m-%d_%H:%M:%S")
    final_grid_df.to_csv(params['valid_result_dir']+ "{}grid_search.csv".format(currentDT_str), index = False)    

