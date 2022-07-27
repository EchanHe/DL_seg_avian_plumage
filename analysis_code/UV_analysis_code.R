## ---------------------------
##
## Script name: UV_analysis_code.R
##
## Purpose of script: Analyse UV of passerine birds
##
## Author: Chris Cooney
## Email: c.cooney@sheffield.ac.uk 


## Phylogenetic heritability

rm(list=ls())

setwd("~/Dropbox/Projects/Current/Bird_colouration/UV_reflectance/")

library(ape)
library(parallel)
library(MCMCglmm)

# ------ #

backbelly <- read.csv("Outputs/UV_data.csv", strings=F)
trees <- read.tree("Outputs/UV_trees.txt")

# ------ #

## Split by sex and view

mback <- backbelly[backbelly$sex=="M" & backbelly$view=="Back",]
mbelly <- backbelly[backbelly$sex=="M" & backbelly$view=="Belly",]

fback <- backbelly[backbelly$sex=="F" & backbelly$view=="Back",]
fbelly <- backbelly[backbelly$sex=="F" & backbelly$view=="Belly",]

rownames(mback) <- mback$species
rownames(mbelly) <- mbelly$species
rownames(fback) <- fback$species
rownames(fbelly) <- fbelly$species

# ------ #

Ainv.list <- as.list(rep(NA, length(trees)))
for (i in 1:length(trees)) {
  Ainv.list[[i]] <- inverseA(trees[[i]], nodes="ALL", scale=T)$Ainv
}

# ------ #

## Tree distribution
## MCMCGLMM - lambda

myFunGauss <- function(x, form, dat) {
  fit <- MCMCglmm(form, random=~phylo, ginverse=list(phylo=Ainv.list[[x]]), data=dat, prior=prior.gauss, verbose=TRUE, nitt=110000, burnin=10000, thin=25)
  return(fit)
}

myFunCat <- function(x, form, dat) {
  fit <- MCMCglmm(form, random=~phylo, ginverse=list(phylo=Ainv.list[[x]]), data=dat, prior=prior.cat, verbose=TRUE, nitt=110000, burnin=10000, thin=25, family="categorical")
  return(fit)
}

myFunSum <- function (fits) {
  sol <- c()
  vcv <- c()
  for (i in 1:length(fits)) {
    sol <- rbind(sol, fits[[i]]$Sol)
    vcv <- rbind(vcv, fits[[i]]$VCV)
  }
  full <- fits[[1]]
  full$Sol <- as.mcmc(sol)
  full$VCV <- as.mcmc(vcv)
  return(full)
}

# ------ #

## Phylogenetic heritability

ntrees <- 100
ncores <- 8

prior.gauss <- list(R=list(V=1, nu=0.002), G=list(G1=list(V=1, nu=0.002)))
prior.cat <- list(R=list(V=1, fix=1), G=list(G1=list(V=1, nu=0.002)))

# From old package MCMCglmmExtras
varComp=function(mod,units=T) { #Tabulates variance components of random effects for MCMCglmm models
  VCV=mod$VCV
  if(units==F) VCV=VCV[,c(1:ncol(VCV)-1)] #Strips out "units" column (for binomial models)
  round(data.frame(mode=posterior.mode(VCV/rowSums(VCV)),HPDinterval(VCV/rowSums(VCV))),3)
}

# mean.u

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.u) ~ 1"), dat=mback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.back.mean.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.u) ~ 1"), dat=fback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.back.mean.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.u) ~ 1"), dat=mbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.belly.mean.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.u) ~ 1"), dat=fbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.belly.mean.u.rds")
full <- myFunSum(fit); varComp(full)

# mean.q50.u

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q50.u) ~ 1"), dat=mback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.back.mean.q50.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q50.u) ~ 1"), dat=fback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.back.mean.q50.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q50.u) ~ 1"), dat=mbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.belly.mean.q50.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q50.u) ~ 1"), dat=fbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.belly.mean.q50.u.rds")
full <- myFunSum(fit); varComp(full)

# mean.q75.u

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q75.u) ~ 1"), dat=mback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.back.mean.q75.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q75.u) ~ 1"), dat=fback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.back.mean.q75.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q75.u) ~ 1"), dat=mbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.belly.mean.q75.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q75.u) ~ 1"), dat=fbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.belly.mean.q75.u.rds")
full <- myFunSum(fit); varComp(full)

# mean.q90.u

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q90.u) ~ 1"), dat=mback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.back.mean.q90.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q90.u) ~ 1"), dat=fback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.back.mean.q90.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q90.u) ~ 1"), dat=mbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.belly.mean.q90.u.rds")
full <- myFunSum(fit); varComp(full)

fit <- mclapply(1:ntrees, myFunGauss, form=formula("log10(mean.q90.u) ~ 1"), dat=fbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.belly.mean.q90.u.rds")
full <- myFunSum(fit); varComp(full)

# presab.uvcol01

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol01 ~ 1"), dat=mback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.back.presab.uvcol01.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3) # https://stat.ethz.ch/pipermail/r-sig-mixed-models/2012q3/019060.html

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol01 ~ 1"), dat=fback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.back.presab.uvcol01.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol01 ~ 1"), dat=mbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.belly.presab.uvcol01.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol01 ~ 1"), dat=fbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.belly.presab.uvcol01.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)

# presab.uvcol05

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol05 ~ 1"), dat=mback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.back.presab.uvcol05.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3) # https://stat.ethz.ch/pipermail/r-sig-mixed-models/2012q3/019060.html

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol05 ~ 1"), dat=fback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.back.presab.uvcol05.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol05 ~ 1"), dat=mbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.belly.presab.uvcol05.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol05 ~ 1"), dat=fbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.belly.presab.uvcol05.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)

# presab.uvcol10

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol10 ~ 1"), dat=mback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.back.presab.uvcol10.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3) # https://stat.ethz.ch/pipermail/r-sig-mixed-models/2012q3/019060.html

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol10 ~ 1"), dat=fback, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.back.presab.uvcol10.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol10 ~ 1"), dat=mbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/m.belly.presab.uvcol10.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)

fit <- mclapply(1:ntrees, myFunCat, form=formula("presab.uvcol10 ~ 1"), dat=fbelly, mc.cores=ncores); saveRDS(fit, "Outputs/heritability_models/f.belly.presab.uvcol10.rds")
full <- myFunSum(fit); tmp <- as.mcmc(full$VCV[,1]/(rowSums(full$VCV)+pi^2/3)); round(data.frame(mode=posterior.mode(tmp), HPDinterval(tmp)), 3)


# ============================== #


## Multipredictor models of UV variables

rm(list=ls())

setwd("~/Dropbox/Projects/Current/Bird_colouration/UV_reflectance/")

library(ape)
library(parallel)
library(MCMCglmm)

# ------ #

backbelly <- read.csv("Outputs/UV_data.csv", strings=F)
trees <- read.tree("Outputs/UV_trees.txt")

# ------ #

Ainv.list <- as.list(rep(NA, length(trees)))
for (i in 1:length(trees)) {
  Ainv.list[[i]] <- inverseA(trees[[i]], nodes="ALL", scale=T)$Ainv
}

# ------ #
## Tree distribution
## MCMCGLMM - lambda

myFun <- function(x, form, dat) {
  fit <- MCMCglmm(form, random=~phylo, ginverse=list(phylo=Ainv.list[[x]]), data=dat, prior=prior, verbose=TRUE, nitt=110000, burnin=10000, thin=25)
  return(fit)
}

myFunCat <- function(x, form, dat) {
  fit <- MCMCglmm(form, random=~phylo, ginverse=list(phylo=Ainv.list[[x]]), data=dat, prior=prior, verbose=TRUE, nitt=110000, burnin=11000, thin=25, family="categorical")
  return(fit)
} 

ntrees <- 100

# ----- #

# mean.u

prior <- list(R=list(V=1, nu=0.002), G=list(G1=list(V=1, nu=0.002)))

form <- formula("mean.u ~ sex.bin + view.bin + uvb + solrad + temp + forest.dep.bin2 + ForStrat.canopy + visual.bin")

backbelly.mod <- mclapply(1:ntrees, myFun, form=form, dat=backbelly, mc.cores=20)
saveRDS(backbelly.mod, "Outputs/predictor_models/mean.u.backbelly.form.rds")

# mean.q50.u

prior <- list(R=list(V=1, nu=0.002), G=list(G1=list(V=1, nu=0.002)))

form <- formula("mean.q50.u ~ sex.bin + view.bin + uvb + solrad + temp + forest.dep.bin2 + ForStrat.canopy + visual.bin")

backbelly.mod <- mclapply(1:ntrees, myFun, form=form, dat=backbelly, mc.cores=20)
saveRDS(backbelly.mod, "Outputs/predictor_models/mean.q50.u.backbelly.form.rds")

# mean.q75.u

prior <- list(R=list(V=1, nu=0.002), G=list(G1=list(V=1, nu=0.002)))

form <- formula("mean.q75.u ~ sex.bin + view.bin + uvb + solrad + temp + forest.dep.bin2 + ForStrat.canopy + visual.bin")

backbelly.mod <- mclapply(1:ntrees, myFun, form=form, dat=backbelly, mc.cores=20)
saveRDS(backbelly.mod, "Outputs/predictor_models/mean.q75.u.backbelly.form.rds")

# mean.q90.u

prior <- list(R=list(V=1, nu=0.002), G=list(G1=list(V=1, nu=0.002)))

form <- formula("mean.q90.u ~ sex.bin + view.bin + uvb + solrad + temp + forest.dep.bin2 + ForStrat.canopy + visual.bin")

backbelly.mod <- mclapply(1:ntrees, myFun, form=form, dat=backbelly, mc.cores=20)
saveRDS(backbelly.mod, "Outputs/predictor_models/mean.q90.u.backbelly.form.rds")

# presab.uvcol01

prior <- list(R=list(V=1, fix=1), G=list(G1=list(V=1, nu=0.002)))

form <- formula("presab.uvcol01 ~ sex.bin + view.bin + uvb + solrad + temp + forest.dep.bin2 + ForStrat.canopy + visual.bin")

backbelly.mod <- mclapply(1:ntrees, myFunCat, form=form, dat=backbelly, mc.cores=20)
saveRDS(backbelly.mod, "Outputs/predictor_models/presab.uvcol01.backbelly.form.rds")

# presab.uvcol05

prior <- list(R=list(V=1, fix=1), G=list(G1=list(V=1, nu=0.002)))

form <- formula("presab.uvcol05 ~ sex.bin + view.bin + uvb + solrad + temp + forest.dep.bin2 + ForStrat.canopy + visual.bin")

backbelly.mod <- mclapply(1:ntrees, myFunCat, form=form, dat=backbelly, mc.cores=20)
saveRDS(backbelly.mod, "Outputs/predictor_models/presab.uvcol05.backbelly.form.rds")


# presab.uvcol10

prior <- list(R=list(V=1, fix=1), G=list(G1=list(V=1, nu=0.002)))

form <- formula("presab.uvcol10 ~ sex.bin + view.bin + uvb + solrad + temp + forest.dep.bin2 + ForStrat.canopy + visual.bin")

backbelly.mod <- mclapply(1:ntrees, myFunCat, form=form, dat=backbelly, mc.cores=20)
saveRDS(backbelly.mod, "Outputs/predictor_models/presab.uvcol10.backbelly.form.rds")


# ============================== #


# Summarise multipredictor model output

rm(list=ls())

library(MCMCglmm)
library(parallel)

setwd("~/Dropbox/Projects/Current/Bird_colouration/UV_reflectance/")

# ------- #

myFunGauss <- function (fits) { # phylo
  sol <- c()
  vcv <- c()
  for (i in 1:length(fits)) {
    sol <- rbind(sol, fits[[i]]$Sol)
    vcv <- rbind(vcv, fits[[i]]$VCV)
  }
  full <- fits[[1]]
  full$Sol <- as.mcmc(sol)
  full$VCV <- as.mcmc(vcv)
  full.out <- summary(full)$solutions
  full.out <- cbind(full.out, t(apply(full$Sol, 2, quantile, probs=c(0.025,0.25,0.5,0.75,0.975))))
  full.out <- data.frame(full.out)
  vmVarF<-numeric(nrow(full$Sol)) # MCMCglmm - marginal (fixed effects) with CI
  for(i in 1:nrow(full$Sol)){
    Var <- var(as.vector(full$Sol[i,] %*% t(full$X)))
    vmVarF[i] <- Var
  }
  R2m <- vmVarF / (vmVarF + full$VCV[,"phylo"] + full$VCV[,"units"])
  full.out$r2m.mean <- mean(R2m)
  full.out$r2m.lci <- HPDinterval(R2m)[1]
  full.out$r2m.uci <- HPDinterval(R2m)[2]
  R2c <- (vmVarF + full$VCV[,"phylo"])/(vmVarF + full$VCV[,"phylo"] + full$VCV[,"units"]) # MCMCglmm - conditional (fixed + random effects) with CI
  full.out$r2c.mean <- mean(R2c)
  full.out$r2c.lci <- HPDinterval(R2c)[1]
  full.out$r2c.uci <- HPDinterval(R2c)[2]
  full.out <- data.frame(full.out, print.est=paste(format(round(full.out[,"post.mean"],3), nsmall=3), " (", format(round(full.out[,"l.95..CI"],3), nsmall=3), ", ", format(round(full.out[,"u.95..CI"],3), nsmall=3), ")", sep=""))
  full.out <- data.frame(full.out, print.p=format(round(full.out[,"pMCMC"],3), nsmall=3))
  return(full.out)
}

myFunCat <- function (fits) { # phylo
  sol <- c()
  vcv <- c()
  for (i in 1:length(fits)) {
    sol <- rbind(sol, fits[[i]]$Sol)
    vcv <- rbind(vcv, fits[[i]]$VCV)
  }
  full <- fits[[1]]
  full$Sol <- as.mcmc(sol)
  full$VCV <- as.mcmc(vcv)
  full.out <- summary(full)$solutions
  full.out <- cbind(full.out, t(apply(full$Sol, 2, quantile, probs=c(0.025,0.25,0.5,0.75,0.975))))
  full.out <- data.frame(full.out)
  c2 <- (16 * sqrt(3)/(15 * pi))^2
  full$Sol1 <- full$Sol/sqrt(1+c2 * full$VCV[, "units"]) 
  full$VCV1 <- full$VCV/(1+c2 * full$VCV[, "units"]) 
  
  vmVarF <- numeric(nrow(full$Sol))
  for(i in 1:nrow(full$Sol)){
    Var <- var(as.vector(full$Sol1[i,] %*% t(full$X)))
    vmVarF[i] <- Var
  }
  R2m <- vmVarF/(vmVarF + full$VCV1[,"phylo"] + pi^2/3)
  full.out$r2m.mean <- mean(R2m)
  full.out$r2m.lci <- HPDinterval(R2m)[1]
  full.out$r2m.uci <- HPDinterval(R2m)[2]
  R2c <- (vmVarF + full$VCV1[,"phylo"])/(vmVarF + full$VCV1[,"phylo"] + pi^2/3)
  full.out$r2c.mean <- mean(R2c)
  full.out$r2c.lci <- HPDinterval(R2c)[1]
  full.out$r2c.uci <- HPDinterval(R2c)[2]
  full.out <- data.frame(full.out, print.est=paste(format(round(full.out[,"post.mean"],3), nsmall=3), " (", format(round(full.out[,"l.95..CI"],3), nsmall=3), ", ", format(round(full.out[,"u.95..CI"],3), nsmall=3), ")", sep=""))
  full.out <- data.frame(full.out, print.p=format(round(full.out[,"pMCMC"],3), nsmall=3))
  return(full.out)
}

# ------- #

porder <- c("(Intercept)",
            "sex.bin",
            "view.bin",
            "uvb",
            "solrad",
            "temp",
            "forest.dep.bin2",
            "ForStrat.canopy",
            "visual.bin")

# ------- #

## output results tables

# mean.u
fits <- readRDS("Outputs/predictor_models/mean.u.backbelly.form.rds"); coefs <- myFunGauss(fits)[porder,]
write.csv(coefs, "Outputs/predictor_models/coefs.mean.u.backbelly.form.csv")

# mean.q50.u
fits <- readRDS("Outputs/predictor_models/mean.q50.u.backbelly.form.rds"); coefs <- myFunGauss(fits)[porder,]
write.csv(coefs, "Outputs/predictor_models/coefs.mean.q50.u.backbelly.form.csv")

# mean.q75.u
fits <- readRDS("Outputs/predictor_models/mean.q75.u.backbelly.form.rds"); coefs <- myFunGauss(fits)[porder,]
write.csv(coefs, "Outputs/predictor_models/coefs.mean.q75.u.backbelly.form.csv")

# mean.q90.u
fits <- readRDS("Outputs/predictor_models/mean.q90.u.backbelly.form.rds"); coefs <- myFunGauss(fits)[porder,]
write.csv(coefs, "Outputs/predictor_models/coefs.mean.q90.u.backbelly.form.csv")

# presab.uvcol01
fits <- readRDS("Outputs/predictor_models/presab.uvcol01.backbelly.form.rds"); coefs <- myFunCat(fits)[porder,]
write.csv(coefs, "Outputs/predictor_models/coefs.presab.uvcol01.backbelly.form.csv")

# presab.uvcol05
fits <- readRDS("Outputs/predictor_models/presab.uvcol05.backbelly.form.rds"); coefs <- myFunCat(fits)[porder,]
write.csv(coefs, "Outputs/predictor_models/coefs.presab.uvcol05.backbelly.form.csv")

# presab.uvcol10
fits <- readRDS("Outputs/predictor_models/presab.uvcol10.backbelly.form.rds"); coefs <- myFunCat(fits)[porder,]
write.csv(coefs, "Outputs/predictor_models/coefs.presab.uvcol10.backbelly.form.csv")


# ======================================================= #
