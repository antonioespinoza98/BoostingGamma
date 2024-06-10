# libraries ---------------------------------------------------------------

x <- c('xgboost','tidyverse','fitdistrplus','data.table')
lapply(x, require, character.only = TRUE)


# read data ---------------------------------------------------------------
rm(list = ls())

base <- read.csv("datos/ehresp_2014.csv") |>
  dplyr::filter(erbmi > 0)


# Análisis de distribución ------------------------------------------------
bmi <- base$erbmi

summary(bmi)

fit <- fitdist(base$erbmi, distr = "gamma")

descdist(bmi, boot=1000)


gammafit  <-  fitdistrplus::fitdist(bmi, "gamma")
weibullfit  <-  fitdistrplus::fitdist(bmi, "weibull")
lnormfit  <-  fitdistrplus::fitdist(bmi, "lnorm")  
gengammafit  <-  fitdistrplus::fitdist(bmi, "gengamma",
                                       start=function(d) list(mu=mean(d),
                                                              sigma=sd(d),
                                                              Q=0))

qqcomp(list(gammafit, weibullfit, gengammafit),
       legendtext=c("gamma","weibull", "gengamma") )

qqcomp(gengammafit)
denscomp(gengammafit)

fitdistr(bmi, "gamma")

