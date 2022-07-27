## ---------------------------
##
## Script name: plot_training_Set_size.R
##
## Purpose of script: Plot IOU, precision and recall to the size of training set used in training
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

df_size = read.csv("data/train_size.csv")

df_size =melt(df_size, id=c("file.vis",'power' , "training_set_size" , 'view' , 'training_set_percentage'))

df_size

df_size_gp_by_pct = df_size %>% group_by(training_set_percentage,variable) %>% 
  summarise(value = mean(value) )


ggplot(data=df_size_gp_by_pct, aes(x=training_set_percentage, y=value, color = variable)) +
  geom_line()+
  geom_point()
