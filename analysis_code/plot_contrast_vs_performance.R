## ---------------------------
##
## Script name: plot_contrast.R
##
## Purpose of script: Plot plots to compare IOU, precision and recall of Deeplab and four classic methods to the image contrast.
##
## Author: Yichen He
## Email: csyichenhe@gmail.com

rm(list=ls())


library(dplyr)
library(ggpubr)
library(ggplot2)
library(reshape)
library(fmsb)
library(RColorBrewer)

df_contrast = read.csv("data/contrast.csv")
  
df_contrast$Method = factor(df_contrast$Method, 
                          levels = c("DeepLabv3+", 'Thresholding' , 'Region Growing' , 'Chan-Vese', 'Graph Cut'))

sp <- ggscatter(df_contrast, x = "contrast", y = "value",alpha = 0.35,
                add = "reg.line",  # Add regressin line
                add.params = list(color = "blue", fill = "lightgray"), # Customize reg. line
                conf.int = TRUE # Add confidence interval
) +  facet_wrap(~ variable + Method , ncol=5)

sp = sp+ stat_cor(method = "pearson", label.x = 0, label.y = 1.5 , size =5) +
  theme(strip.text = element_text(size = 16))

sp
