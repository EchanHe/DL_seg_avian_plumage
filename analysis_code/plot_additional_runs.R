## ---------------------------
##
## Script name: plot_additional_runs
##
## Purpose of script: Plot plots to compare IOU, precision and recall of addtional runs:
##                    Including: resolutions, input channels, image augmentation, model subsetting
##
## Author: Yichen He
## Email: csyichenhe@gmail.com

rm(list=ls())



library("plyr") 
library(dplyr)
library(ggplot2)
library(ggpubr)
library(gridExtra)


cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


###### Resolution

#### resolution boxplots

# read the evalution results of resolution
df_reso = read.csv("data/resolutions.csv")


plot_list <- list()
for (metric in c("IOU" , "Precision" , "Recall")){
  p <- ggboxplot(df_reso, x = "Resolution", y = metric, palette = cbPalette)
  #  Add p-value
  p=p + # stat_compare_means(method = "anova",size=8)+
    stat_compare_means(label = "p.signif", method = "t.test",
                       ref.group = '618x410' ,size=7)
  
  p=p+font("xlab", size = 24)+font("ylab", size = 24)+
    
    theme(strip.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 22),
          axis.text.x = element_text(size = 20))
  plot_list = append(plot_list ,list(p))
  
}
p=do.call("grid.arrange", c(plot_list, ncol=3))

# ggsave("plots/reso.tiff",p , width = 17, height=5)



#### Resoluion plot Tuckey test

df_tuckey_all = data.frame()
for (metric in c("IOU" , "Precision" , "Recall")){

  model=lm( df_reso[[metric]] ~ df_reso$Resolution )
  ANOVA=aov(model)
  

  TUKEY <- TukeyHSD(x=ANOVA, 'df_reso$Resolution', conf.level=0.95)
  df_TUKEY = data.frame(TUKEY$`df_reso$Resolution`)
  df_TUKEY$is_signif= (df_TUKEY$lwr * df_TUKEY$upr) >0
  df_TUKEY$group = rownames(df_TUKEY)
  df_TUKEY$Metric = metric
  
  df_tuckey_all = rbind.fill(df_tuckey_all , df_TUKEY)
}


df_tuckey_all$group = factor(df_tuckey_all$group   , levels = rev(unique(df_tuckey_all$group)))

p=ggplot(df_tuckey_all, aes(color = is_signif) )+
  geom_point( aes(y=group , x=diff),size = 4)+
  geom_linerange(aes(xmin = lwr, xmax = upr , y=group),size = 1)+geom_vline(xintercept=0, linetype="dotted", color = "red")+
  
  facet_wrap(~ Metric, ncol=3) +
  scale_color_manual(values=c("#999999" , "#56B4E9"))


p=p+theme_classic()+
  theme(strip.text.x =  element_text(size = 18),
        axis.text.y = element_text(size = 18),
        axis.text.x = element_text(size = 18),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        legend.position = 'None') 

p



###### Input Channels

#### Input Channels box plots
df_uv=read.csv("data/UV.csv")

plot_list <- list()
for (metric in c("IOU" , "Precision" , "Recall")){
  p <- ggboxplot(df_uv, x = "Channels", y = metric, palette = cbPalette)
  #  Add p-value
  p=p + # stat_compare_means(method = "anova",size=8)+
    stat_compare_means(label = "p.signif", method = "t.test",
                       ref.group = 'RGB' ,size=7)
  
  p=p+font("xlab", size = 24)+font("ylab", size = 24)+
    
    theme(strip.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 22),
          axis.text.x = element_text(size = 20))
  plot_list = append(plot_list ,list(p))
  
}
p=do.call("grid.arrange", c(plot_list, ncol=3))
p

#### Input Channels tukey test plots

df_tuckey_all = data.frame()
for (metric in c("IOU" , "Precision" , "Recall")){

  model=lm( df_uv[[metric]] ~ df_uv$Channels )
  ANOVA=aov(model)

  TUKEY <- TukeyHSD(x=ANOVA, 'df_uv$Channels', conf.level=0.95)
  df_TUKEY = data.frame(TUKEY$`df_uv$Channels`)
  df_TUKEY$is_signif= (df_TUKEY$lwr * df_TUKEY$upr) >0
  df_TUKEY$group = rownames(df_TUKEY)
  df_TUKEY$Metric = metric
  
  df_tuckey_all = rbind.fill(df_tuckey_all , df_TUKEY)
}


df_tuckey_all$group = factor(df_tuckey_all$group   , levels = rev(unique(df_tuckey_all$group)))

p=ggplot(df_tuckey_all, aes(color = is_signif) )+
  geom_point( aes(y=group , x=diff),size = 4)+
  geom_linerange(aes(xmin = lwr, xmax = upr , y=group),size = 1)+geom_vline(xintercept=0, linetype="dotted", color = "red")+
  theme_classic()+facet_wrap(~ Metric, ncol=3) + scale_color_manual(values=c("#56B4E9", "#999999" ))


p=p+
  theme(strip.text.x =  element_text(size = 18),
        axis.text.y = element_text(size = 18),
        axis.text.x = element_text(size = 18),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        legend.position = 'None') 

p



###### image augmentation 

#### image augmentation box plots

df_aug= read.csv("data/aug.csv")

plot_list <- list()
for (metric in c("IOU" , "Precision" , "Recall")){
  p <- ggboxplot(df_aug, x = "Data.augmentation", y = metric, palette = cbPalette )
  #  Add p-value
  p=p + # stat_compare_means(method = "anova",size=8)+
    stat_compare_means(label = "p.signif", method = "t.test",alternative = "greater",
                       ref.group = 'NO' ,size=7)
  
  p=p+font("xlab", size = 24)+font("ylab", size = 24)+
    
    theme(strip.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 22),
          axis.text.x = element_text(size = 20))
  plot_list = append(plot_list ,list(p))
  
  
  
}
p=do.call("grid.arrange", c(plot_list, ncol=3))
p

t.test(IOU ~ Data.augmentation, data = df_aug, var.equal=TRUE)

t.test(Precision ~ Data.augmentation, data = df_aug, var.equal=TRUE)
t.test(Recall ~ Data.augmentation, data = df_aug, var.equal=TRUE)


###### Model subetting

#### Model subetting boxplots
df_subset = read.csv("data/subsetting.csv")
plot_list <- list()
for (metric in c("IOU" , "Precision" , "Recall")){
  p <- ggboxplot(df_subset, x = "Model.subsetting", y = metric, palette = cbPalette )
  #  Add p-value
  p=p + # stat_compare_means(method = "anova",size=8)+
    stat_compare_means(label = "p.signif", method = "t.test",
                       ref.group = 'NO' ,size=7)
  
  p=p+font("xlab", size = 24)+font("ylab", size = 24)+
    
    theme(strip.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 22),
          axis.text.x = element_text(size = 20))
  plot_list = append(plot_list ,list(p))
  
}
p=do.call("grid.arrange", c(plot_list, ncol=3))
p



plot_list <- list()
for (metric in c("IOU" , "Precision" , "Recall")){
  p <- ggboxplot(df_subset, x = "Model.subsetting",facet.by='view', y = metric, palette = cbPalette)
  #  Add p-value
  p=p + # stat_compare_means(method = "anova",size=8)+
    stat_compare_means(label = "p.signif", method = "t.test",
                       ref.group = 'NO' ,size=7)
  
  p=p+font("xlab", size = 24)+font("ylab", size = 24)+
    
    theme(strip.text.x = element_text(size = 22),
          axis.text.y = element_text(size = 22),
          axis.text.x = element_text(size = 20))
  plot_list = append(plot_list ,list(p))
  
}
p=do.call("grid.arrange", c(plot_list, ncol=3))
p


t.test(IOU ~ Model.subsetting, data = df_subset,var.equal=TRUE)
t.test(Precision ~ Model.subsetting, data = df_subset,var.equal=TRUE)
t.test(Recall ~ Model.subsetting, data = df_subset,var.equal=TRUE)

df_back = df_subset[df_subset$view=="back",]

t.test(IOU ~ Model.subsetting, data = df_back,var.equal=TRUE)
t.test(Precision ~ Model.subsetting, data = df_back,var.equal=TRUE)
t.test(Recall ~ Model.subsetting, data = df_back,var.equal=TRUE)


df_belly = df_subset[df_subset$view=="belly",]

t.test(IOU ~ Model.subsetting, data = df_belly,var.equal=TRUE)
t.test(Precision ~ Model.subsetting, data = df_belly,var.equal=TRUE)
t.test(Recall ~ Model.subsetting, data = df_belly,var.equal=TRUE)

df_side = df_subset[df_subset$view=="side",]

t.test(IOU ~ Model.subsetting, data = df_side,var.equal=TRUE)
t.test(Precision ~ Model.subsetting, data = df_side,var.equal=TRUE)
t.test(Recall ~ Model.subsetting, data = df_side,var.equal=TRUE)
