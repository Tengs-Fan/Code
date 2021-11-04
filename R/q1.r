# Input Age
age<-c(23,23,27,27,39,41,47,49,50,52,54,54,56,57,58,58,60,61)
# Input %fat
fat<-c(9.5,26.5,7.8,17.8,31.4,25.9,27.4,27.2,31.2,34.6,42.5,28.8,33.4,30.2,34.1,32.9,41.2,35.7)

# 1 Calculation
#Mean
mean(age); mean(fat)
#Median
median(age); median(fat)
#Standard Deviation
sd(age); sd(fat)

# 2 Box Plot
boxplot(age,xlab = "age",ylab="value",main="Age Box Plot")
boxplot(fat,xlab = "fat",ylab="value",main="Fat Box Plot")

# 3.1 Scatter Plot
plot(age,fat, type="p",main="Scatter Plot") 
# 3.2 Q-Q Plot
qqnorm(age);qqline(age)
qqnorm(fat);qqline(fat)
