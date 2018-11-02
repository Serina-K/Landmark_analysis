library(tictoc)
library(polycor)

# Continuous  	Ordinal 	N 	Correlation 
# Age 	Oxygen 	29 	-0.23586 	
# Weight 	Oxygen 	29 	-0.24514 	
# RunTime 	Oxygen 	28 	-0.91042 

# clear console
cat("\014")  
#setwd('/home/zeynep/Dropbox/research/supervision/2018_04_koyama/landmark_analysis_v2/polyserial_example/')
setwd('/Users/zeynep/Dropbox/research/supervision/2018_04_koyama/landmark_analysis_v2/polyserial_example/')
mydata <- read.table(file="sample_data_v3.txt",  sep="\t", dec=",",  na.strings=c("."))

age = mydata$V1
weight = mydata$V2
runtime = mydata$V3
oxygen = mydata$V4

tic("Time ")
p1 <- polyserial(age, oxygen, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("Age-oxygen: ", p1))

p2 <- polyserial(weight, oxygen, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("Weight-oxygen: ", p2))

p3 <- polyserial(runtime, oxygen, ML = TRUE, control = list(), std.err = FALSE, maxcor=.999999, bins=6)
print(paste0("Runtime-oxygen: ", p3))

toc()
