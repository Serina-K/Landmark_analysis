library(tictoc)
library(polycor)

# Continuous  	Ordinal 	N 	Correlation 
# Age 	Oxygen 	29 	-0.23586 	
# Weight 	Oxygen 	29 	-0.24514 	
# RunTime 	Oxygen 	28 	-0.91042 

# clear console
cat("\014")  

data1 = read.table("annotator.csv",  sep="\t", dec=".",  na.strings=c("."))
data2 = read.table("eyesize.csv",  sep="\t", dec=".",  na.strings=c("."))
data3 = read.table("LM_roll.csv",  sep="\t", dec=".",  na.strings=c("."))
data4 = read.table("ACC_roll_pitch_yaw.csv",  sep="\t", dec=".",  na.strings=c("."))
data5 = read.table("eyedist.csv",  sep="\t", dec=".",  na.strings=c("."))
data6 = read.table("frame_number_blink.csv",  sep="\t", dec=".",  na.strings=c("."))
data7 = read.table("mean_open.csv",  sep="\t", dec=".",  na.strings=c("."))
data8 = read.table("number_blink.csv",  sep="\t", dec=".",  na.strings=c("."))
data9 = read.table("db.csv",  sep="\t", dec=".",  na.strings=c("."))

#read.table->read.delim
#dec=","->dec="."




annotator1 = data1$V1
annotator2 = data1$V2
eyesize = data2$V1
LM_roll = data3$V1
ACC_roll = data4$V1
ACC_pitch = data4$V2
ACC_yaw = data4$V3

ano1 = data5$V1
ano2 = data5$V2
eyedist1 = data5$V3
eyedist2 = data5$V4

Rb = data6$V1
Aopen = data7$V1

nb = data8$V1
db = data9$V1



tic("Time ")
p1 <- polyserial(eyesize, annotator1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_eyesize_1: ", p1))

p2 <- polyserial(eyesize, annotator2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_eyesize_2: ", p2))

p3 <- polyserial(LM_roll, annotator1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_LM_roll_1: ", p3))

p4 <- polyserial(LM_roll, annotator2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_LM_roll_2: ", p4))

p5 <- polyserial(ACC_roll, annotator1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_ACC_roll_1: ", p5))

p6 <- polyserial(ACC_roll, annotator2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_ACC_roll_2: ", p6))

p7 <- polyserial(ACC_pitch, annotator1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_ACC_pitch_1: ", p7))

p8 <- polyserial(ACC_pitch, annotator2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_ACC_pitch_2: ", p8))

p9 <- polyserial(ACC_yaw, annotator1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_ACC_yaw_1: ", p9))

p10 <- polyserial(ACC_yaw, annotator2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("cor_ACC_yaw_2: ", p10))


p11 <- polyserial(eyedist1, ano1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("eyedist1_ano1: ", p11))

p12 <- polyserial(eyedist1, ano2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("eyedist1_ano2: ", p12))

p13 <- polyserial(eyedist2, ano1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("eyedist2_ano1: ", p13))

p14 <- polyserial(eyedist2, ano2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("eyedist2_ano2: ", p14))




p15 <- polyserial(Rb, ano1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("Rb_ano1: ", p15))

p16 <- polyserial(Rb, ano2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("Rb_ano2: ", p16))



p17 <- polyserial(Aopen, ano1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("Aopen_ano1: ", p17))

p18 <- polyserial(Aopen, ano2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("Aopen_ano2: ", p18))


p19 <- polyserial(nb, ano1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("nb_ano1: ", p19))

p20 <- polyserial(nb, ano2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("nb_ano2: ", p20))

p21 <- polyserial(db, ano1, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("db_ano1: ", p21))

p22 <- polyserial(db, ano2, ML = TRUE, control = list(), std.err = FALSE, maxcor=.9999, bins=4)
print(paste0("db_ano2: ", p22))


toc()
