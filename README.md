# DL_seg_avian_plumage

This is the repo for the analysis code in **Deep learning image segmentation reveals patterns of UV reflectance evolution in passerine birds** (He et. al. 2022).

This is the code using DeepLab to segment plumage area of bird images and compare the performance to some classic segmentation methods (thresholding, region growing, graph cut and chan-vese). The model is applied on a dataset of passerine birds. The segmentation results are then used to measure the UV reflectance which is used to analyse the evolution of UV in passerine birds.

## Prerequisites
- Python 3
- tensorflow = 1.6.0
- numpy >= 1.17.3
- pandas >= 0.23.4
- opencv-python = 4.1.1.26
- scikit-imgae = TODO

- R = 4.1.0
- TODO r packages

## Installation

```bash

git clone https://github.com/EchanHe/DL_seg_avian_plumage.git
```

## Usage
Segmentation codes are stored in `segmentation_code/` folder.
5 images and their annotations are stored in `segmentation_code/data` folder, the `visualise_data.ipynb` can be used to visualise the segmentation.

Classic segmentation methods are implemented in `classic_segmentation_methods.ipynb`

#### Config files
Adjust the `[Directory]` section to fit your workspace

`file_col` is the column name for the file name

`cols_override` is the column names for segmentation

The other settings and hyperparameters can be set in the config file as well.

#### Training

```python
python train.py <training_config>.cfg
```

#### Predicting
```python
python pred.py <predict_config>.cfg
```

Analysis codes are stored in `analysis_code/` folder including codes for plotting figures in the paper and analyse the UV reflectance of passerine birds.
