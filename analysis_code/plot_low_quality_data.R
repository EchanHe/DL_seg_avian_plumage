## ---------------------------
##
## Script name: plot_low_quality_data.R
##
## Purpose of script: Plot plots to compare IOU, precision and recall of the original dataset and four low quality datasets
##
## Author: Yichen He
## Email: csyichenhe@gmail.com


rm(list=ls())

library(dplyr)
library(ggplot2)
library(ggpubr)
library(gridExtra)


df_data=read.csv("data/low_quali_data.csv")

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

#### box plots

plot_list <- list()
for (metric in c("IOU" , "Precision" , "Recall")){
  p <- ggboxplot(df_data, x = "Dataset", y = metric ,
                 color = "Dataset" , palette = cbPalette,xlab = FALSE )
  #  Add p-value
  p=p + # stat_compare_means(method = "anova",size=8)+
    stat_compare_means(label = "p.signif", method = "t.test",
                       ref.group = "Origin" ,size=8)
  
  p=p+font("xlab", size = 24)+font("ylab", size = 24)+
    theme(strip.text.x = element_text(size = 20),
          axis.text.y = element_text(size = 22),
          legend.title = element_text(size = 22),
          legend.text = element_text(size = 20),
          axis.text.x = element_blank(),
          legend.margin=margin(),
          legend.position = "none")
  
  plot_list = append(plot_list ,list(p))
  
}
p=do.call("grid.arrange", c(plot_list, ncol=3))
p




#### tukey test plots
df_tuckey_all = data.frame()
for (metric in c("IOU" , "Precision" , "Recall")){
  model=lm( df_data[[metric]] ~ df_data$Dataset )
  ANOVA=aov(model)
  TUKEY <- TukeyHSD(x=ANOVA, 'df_data$Dataset', conf.level=0.95)
  df_TUKEY = data.frame(TUKEY$`df_data$Dataset`)
  df_TUKEY$is_signif= (df_TUKEY$lwr * df_TUKEY$upr) >0
  df_TUKEY$group = rownames(df_TUKEY)
  df_TUKEY$Metric = metric
  
  df_tuckey_all = rbind.fill(df_tuckey_all , df_TUKEY)
}


df_tuckey_all$group = factor(df_tuckey_all$group   , levels = rev(unique(df_tuckey_all$group)))

p=ggplot(df_tuckey_all, aes(color = is_signif) )+
  geom_point( aes(y=group , x=diff),size = 3)+
  geom_linerange(aes(xmin = lwr, xmax = upr , y=group),size = 1)+geom_vline(xintercept=0, linetype="dotted", color = "red")+
  theme_classic()+facet_wrap(~ Metric, ncol=3) + scale_color_manual(values=c("#999999" , "#56B4E9"))


p=p+
  theme(strip.text.x =  element_text(size = 18),
        axis.text.y = element_text(size = 18),
        axis.text.x = element_text(size = 18),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        legend.position = 'None') 

p
