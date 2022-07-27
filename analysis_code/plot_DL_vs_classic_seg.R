## ---------------------------
##
## Script name: plot_DL_vs_classic_seg.R
##
## Purpose of script: Plot plots to compare IOU, precision and recall of Deeplab and four classic methods
##
## Author: Yichen He
## Email: csyichenhe@gmail.com



df = read.csv("data/dl_vs_classic.csv")
# Compute the analysis of variance
res.aov <- aov(value ~ Method, data = df)
# Summary of the analysis
summary(res.aov)


p=ggboxplot(df, x = "Method", y = "value", 
          color = "Method", palette = c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442"),
          order = c("DeepLab","Thresholding","Chan Vese", "Graph cut","Region growing"),
          ylab="Metrics", xlab="")+
  scale_x_discrete(labels=c("", "" ,"","",""))+
  facet_wrap(~ variable, ncol = 3)

p


p + stat_compare_means(method = "anova", label.y = 40)+      # Add global p-value
   stat_compare_means(label = "p.signif", method = "t.test",
                     ref.group = "DeepLab")    

p+stat_compare_means(label = "p.signif", method = "t.test",
                     ref.group = "DeepLab")    
