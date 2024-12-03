## INTRODUCTION ----

# Setting working directory
setwd(choose.dir())

# Clear Environment
rm(list = ls())

install.packages("forecast")
library(forecast)
library(imputeTS)
library(tseries)
library(ggplot2)

# Display numbers in natural form
options(scipen=999)

# Set seed for reproducibility
set.seed(467)

## DATA CLEANING ----

# Read the data
data <- read.csv("PCE.csv", header = TRUE)

# Check data type
str(data)

# Update date column to Date format
data$DATE <- as.Date(data$DATE, format = "%m/%d/%Y")

# Recheck data type
str(data)

# Create the Time Series object from the data set
dataTS <- ts(data$PCE, start=c(1959, 1), end=c(2023, 11), frequency=12)

# Print and plot the Time Series object
dataTS
plot(dataTS, main= "PCE Time Series Plot")

# Check for missing values in the Time Series object and plot distribution
sum(is.na(dataTS))
ggplot_na_distribution2(dataTS, interval_size = 12, 
                        title = "Distribution of 43 Missing Values")

# Impute the missing data
datacomplete <- na_kalman(dataTS)
sum(is.na(datacomplete))

# Create a new time series object by combining all time series objects
ComboDataTS <- cbind(dataTS,datacomplete)

# Print & plot the combined Time Series object
ComboDataTS
colnames(ComboDataTS) <- c("Original_Data", "Imputed_Data")
plot(ComboDataTS, main= "Plot of Time Series both before & after imputation")

# Decompose the time series to understand data pattern
de_add <- decompose(datacomplete, type='additive')
plot(de_add)

# Plot data with trend and seasonal components
plot(datacomplete)
lines(de_add$trend, col=2)
lines(de_add$seasonal, col=3)

## DATA EXPLORATION ----

# Plot the histogram of the Time Series to check data distribution
hist(datacomplete, main = "Histogram of PCE", xlab = "PCE", col = "skyblue")

# Plot the Time Series object and check for any trend
plot(datacomplete, main = "Time Series Plot") # There is a linear relationship

# Plot the seasonal distribution of the Time Series
boxplot(split(datacomplete, cycle(datacomplete)), 
        names = month.abb, col = "skyblue",
        main = "Seasonal Distribution of the Time Series")

# Check Seasonal Plot of the Time Series
seasonplot(datacomplete, main = "Seasonal Plot of the Time Series")

# Plot the time series autocorrelation plot
dev.off()
par(mar = c(4,4,2,2))
tsdisplay(datacomplete, main = "Display of Time Series") # strong autocorrelation indicating non-stationarity

# Check for stationarity
adf.test(datacomplete) # as p-value is greater than 0.05, hence the series is non-stationary
kpss.test(datacomplete) # as p-value is less than 0.05, hence the series is non-stationary

# Converting Non-Stationary into Stationary by taking the 1st difference
diff_data <- diff(datacomplete)
adf.test(diff_data) # p-value is less than 0.05
kpss.test(diff_data) # p-value still less than 0.05, hence need to take 2nd difference

# Converting Non-Stationary into Stationary by taking the 2nd difference
diff2_data <- diff(diff(datacomplete))
adf.test(diff2_data) # Confirmed that new TS is stationary
kpss.test(diff2_data) # Confirmed that new TS is stationary
plot(diff2_data, main = "Time Series with Second Difference", 
     xlab = "Time", ylab = "PCE")

# Plot the time series autocorrelation plot for the differenced time series
tsdisplay(diff2_data, main = "Display of Time Series with Second Difference")

# Smoothen the data - Compute a moving average and plot it against the differenced time series
diffdataMA <- ma(diff2_data, order = 3)
plot(diff2_data, main = "Differenced Time Series with Moving Average", ylab="PCE")
lines(diffdataMA, col = "red")


## DATA MODELLING ----

# Split the data into training and test sets
endmonth <- c(2010, 12)
startmonth <- c(2011, 1)
train <- window(datacomplete, end = endmonth)
test <- window(datacomplete, start = startmonth)
length(test)/(length(datacomplete))*100 # Data has been split around 80-20, 80% data is for training models, and 20% data is for testing

## APPLYING STATIC MODELS ----

### Drift Model ----
par(mar = c(2,2,2,2))
drift_fc <- rwf(train, h = length(test), drift = TRUE)
drift_acc <- accuracy(drift_fc, test)
summary(drift_fc)
plot(drift_fc, main = "Forecasts from Drift Model")
checkresiduals(drift_fc)

### Holt's Linear Model ----
holt_fc <- holt(train, h = length(test))
holt_acc <- accuracy(holt_fc, test)
summary(holt_fc)
plot(holt_fc, main = "Forecasts from Holt's Linear Model")
checkresiduals(holt_fc)

### ARIMA Model ----

# Fit the ARIMA(2,2,2) model
arima_model_1 <- Arima(train, order = c(2, 2, 2))
summary(arima_model_1)
arima_fc_1 <- forecast(arima_model_1, h = length(test))
plot(arima_fc_1, main = "Forecasts from ARIMA(2,2,2) Model")
checkresiduals(arima_fc_1)
arima_acc_1 <- accuracy(arima_fc_1, test)
arima_acc_1

# AUTO ARIMA MODEL
arima_model <- auto.arima(train)
summary(arima_model)
arima_fc <- forecast(arima_model, h = length(test))
plot(arima_fc, main = "Forecasts from ARIMA(3,2,2) Model")
checkresiduals(arima_fc)
arima_acc <- accuracy(arima_fc, test)
arima_acc

## PLOTTING THE MODELS ----

# Plot the original data along with all model forecasts
par(mfrow = c(2, 2), mar = c(2, 2, 2, 2) + 0.1)
plot(datacomplete, main="Original Dataset")
plot(drift_fc, main="Forecasts from Drift Model")
plot(holt_fc, main="Forecasts from Holt's Linear Model")
plot(arima_fc, main="Forecasts from ARIMA(3,2,2) Model")
par(mfrow = c(1, 1))

autoplot(datacomplete, series = "Actual Data", col = "black", main = "Real vs Static Forecast Plot", ylab = "PCE") + 
  autolayer(drift_fc$mean, series = "Drift") +
  autolayer(holt_fc$mean, series = "Holt") +
  autolayer(arima_fc$mean, series = "ARIMA") +
  theme(legend.position = "top", 
        legend.title = element_blank(), 
        plot.title = element_text(hjust = 0.5))


## EVALUATING MODELS ----

# Extract and print accuracy metrics
result <- rbind(drift_acc, holt_acc, arima_acc)
rownames(result) <- c("Drift - Train", "Drift - Test", "Holt - Train", "Holt - Test", "ARIMA - Train", "ARIMA - Test")
result

# save data
write.csv(result, file = "Data/Total_Accuracy.csv", row.names = TRUE)

## DATA FORECASTING ----

# Fit Holt's Linear model using the entire dataset and forecasting 11 months into the future (October 2024)
holt_model_final <- holt(datacomplete, h = 11)
summary(holt_model_final)
plot(holt_model_final)
holt_model_final

# Extract & print the forecasted value for October 2024
Oct_2024_fc <- holt_model_final$mean[length(holt_model_final$mean)]
cat("The PCE forecast for October 2024 is:", Oct_2024_fc, "\n")

## ROLLING FORECASTING ----

### Rolling Drift ----
drift_fit_model <- rwf(train)
drift_refit_model <- rwf(datacomplete, model = drift_fit_model) #refit model to complete dataset with same parameters
drift_roll_fc <- window(fitted(drift_refit_model), start = startmonth) #perform rolling forecast
drift_roll_acc <- accuracy(drift_roll_fc, test)

### Rolling Holt ----
holt_fit_model <- holt(train)
holt_refit_model <- holt(datacomplete, model = holt_fit_model) #refit model to complete dataset with same parameters
holt_roll_fc <- window(fitted(holt_refit_model), start = startmonth) #perform rolling forecast
holt_roll_acc <- accuracy(holt_roll_fc, test)

### Rolling ARIMA----
arima_fit_model <- auto.arima(train)
arima_refit_model <- Arima(datacomplete, model = arima_fit_model) #refit model to complete dataset with same parameters
arima_roll_fc <- window(fitted(arima_refit_model), start = startmonth) #perform rolling forecast
arima_roll_acc <- accuracy(arima_roll_fc, test)

## EVALUATING ROLLING MODELS ----

# Plotting rolling forecasts
autoplot(datacomplete, series = "Actual Data", col = "black", main = "Real vs Rolling Forecast Plot", ylab = "PCE") + 
  autolayer(arima_roll_fc, series = "Rolling ARIMA") +
  autolayer(drift_roll_fc, series = "Rolling Drift") +
  autolayer(holt_roll_fc, series = "Rolling Holt") +
  theme(legend.position = "top", 
        legend.title = element_blank(), 
        plot.title = element_text(hjust = 0.5))

# Plotting all forecasts
autoplot(datacomplete, series = "Actual Data", col = "black", main = "Static vs Rolling Forecasts", ylab = "PCE") + 
  autolayer(drift_fc$mean, series = "DRIFT") +
  autolayer(drift_roll_fc, series = "Rolling DRIFT") +  
  autolayer(holt_fc$mean, series = "HOLT") +
  autolayer(holt_roll_fc, series = "Rolling HOLT") +
  autolayer(arima_fc$mean, series = "ARIMA") +
  autolayer(arima_roll_fc, series = "Rolling ARIMA") +
  theme(legend.position = "top", 
        legend.title = element_blank(), 
        plot.title = element_text(hjust = 0.5))

result_roll <- rbind(drift_roll_acc, holt_roll_acc, arima_roll_acc)
rownames(result_roll) <- c("Rolling DRIFT", "Rolling HOLT","Rolling ARIMA")
result_roll

## COMPARING STATIC & ROLLING FORECASTS ----
# Extracting only test set measures from normal forecast models
drift_test_acc <- c(drift_acc["Test set", ])
holt_test_acc <- c(holt_acc["Test set", ])
arima_test_acc <- c(arima_acc["Test set", ])

result_old <- matrix(c(drift_test_acc, holt_test_acc, arima_test_acc), nrow = 3, byrow = TRUE)
row.names(result_old) <- c("DRIFT", "HOLT", "ARIMA")
col_names <- c("ME", "RMSE", "MAE", "MPE", "MAPE", "MASE", "ACF1", "Theil's U")
colnames(result_old) <- col_names
result_old

# Combine both and evaluate relevant measures (RMSE, MAE, MAPE)
result_roll <- result_roll[,-c(1,4,6,7)]
result_old <- result_old[,-c(1,4,6,7,8)]
result_final <- rbind(result_old, result_roll)
result_final

# save data
write.csv(result_final, file = "Data/Final_Accuracy.csv", row.names = TRUE)