## Purpose of script: Read cfg file for segmenting.
##
## Author: Yichen He
## Date: 2022/07
## Email: csyichenhe@gmail.com

import numpy as np
import pandas as pd

# coords_cols = ['s02.standard_x', 's02.standard_y', 's20.standard_x', 's20.standard_y',
# 's40.standard_x', 's40.standard_y', 's80.standard_x', 's80.standard_y',
# 's99.standard_x', 's99.standard_y','crown_x', 'crown_y', 'nape_x',
# 'nape_y', 'mantle_x', 'mantle_y', 'rump_x', 'rump_y', 'tail_x',
# 'tail_y', 'throat_x', 'throat_y', 'breast_x', 'breast_y', 'belly_x',
# 'belly_y', 'tail.underside_x', 'tail.underside_y', 'wing.coverts_x',
# 'wing.coverts_y', 'wing.primaries.secondaries_x',
# 'wing.primaries.secondaries_y']

coords_cols = ['s02.standard_x', 's02.standard_y', 's20.standard_x', 's20.standard_y',
's40.standard_x', 's40.standard_y', 's80.standard_x', 's80.standard_y',
's99.standard_x', 's99.standard_y','crown_x', 'crown_y', 'nape_x',
'nape_y', 'mantle_x', 'mantle_y', 'rump_x', 'rump_y', 'tail_x',
'tail_y', 'throat_x', 'throat_y', 'breast_x', 'breast_y', 'belly_x',
'belly_y', 'coverts_x', 'coverts_y', 'flight_feathers_x',
'flight_feathers_y']

patches_cols = ['poly.crown' , 'poly.nape','poly.mantle', 'poly.rump', 'poly.tail',
    'poly.throat', 'poly.breast', 'poly.belly', 'poly.tail.underside',
     'poly.wing.coverts',   'poly.wing.primaries.secondaries']

def return_patches_cols(view = 'all',train_ids = None):
    """
    Return the different column names

    """

    patches_cols_np = np.array(patches_cols)



    if view == 'all':
        idx = range(0, len(patches_cols_np))
    if view =='back':
        idx = range(0,5)
    if view =='belly':
        idx = range(5,9)
    if view == 'side':
        idx = range(9,1)
    if train_ids is not None:
        idx = np.intersect1d(idx, train_ids)
    return patches_cols_np[idx]


def return_coords_cols(view = 'all',no_standard = False, train_ids = None):
    """
    Return the different column names

    """
    coords_cols_np = np.array(coords_cols)


    if no_standard:
        idx_stand = range(0,0)
    else:
        idx_stand = range(0, 10)

    if view == 'all' or view is None:
        idx = range(10, len(coords_cols_np))
    if view =='back':
        idx = range(10,20)
    if view =='belly':
        idx = range(20,26)
    if view == 'side':
        idx = range(26,30)

    idx = np.union1d(idx, idx_stand)

    if train_ids is not None:       
        train_ids = np.array(train_ids)
        train_ids = np.stack([2 * train_ids, 2 * train_ids + 1], axis=1).ravel()
        idx = np.intersect1d(idx, train_ids)
    idx = idx.astype(int)
    return coords_cols_np[idx]


