# Input Age
age<-c(13,15,16,16,19,20,20,21,22,22,25,25,25,25,30,33,33,35,35,35,35,36,40,45,46,52,70)

# 1 Min-Max Normalization
norm <- (35 - min(age)) / (max(age) - min(age))

# 2 z-Score Normalizatio
norm <- (35-mean(age)) / 12.94

# 3 Decimal Scaling 
scale <- ceiling(log(max(abs(age)),10))
value <- 35/10^scale
