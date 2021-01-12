
library(dplyr)
library(ggplot2)
library(reticulate)
np = import('numpy')

d = np$load('5_pgdl_pretrain/out/simple_lstm_dl_da_iter.npz')

obs = d$f[['obs']] #[,,1:10]
obs_withheld = d$f[['obs_orig']] # if we withhold observations from DA steps
Y = d$f[['Y']]#[,1:10,]
R = d$f[['R']]#[,,1:10]
Q = d$f[['Q']]
P = d$f[['P']]
n_en = 100
dates = d$f[['dates']]
n_step = length(dates)
cur_model_idxs = d$f[['model_locations']]
n_segs = length(cur_model_idxs)
#preds_no_da = d$f[['preds_no_da']]
Y_no_da = d$f[['Y_no_da']]
Y_forecast = d$f[['Y_forecasts']]
f_horizon = 3 # make dynamic from output
true = d$f[['true']]

add_text <- function(text, location="topright"){
  legend(location,legend=text, bty ="n", pch=NA)
}

#lordville site is seg_id_nat == 1573; model_idx = 224; 1574 & 1575 are directly upstream of 1573 and 1577 is directly downstream of 1573
# cur_model_idxs = '1573'
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc = which(cur_model_idxs == j)
  mean_pred = rowMeans(Y[matrix_loc,,])
  #mean_pred_no_da = rowMeans(Y_no_da[matrix_loc,,]) # colMeans(preds_no_da[,,matrix_loc])

  temp_rmse = round(sqrt(mean((mean_pred - obs[matrix_loc,1,])^2, na.rm = T)), 2)
  #temp_rmse_no_da = round(sqrt(mean((mean_pred_no_da - obs[matrix_loc,1,])^2, na.rm = T)), 2)

  windows(width = 14, height = 10)
  #par(mar = c(3,6,4,3), mfrow = c(2,1))
  plot(Y[matrix_loc,,1] ~ dates, type = 'l',
       ylab = 'Stream Temp (C)', xlab = '', lty=0,
       ylim = c(0,25), #ylim =range(c(Y[matrix_loc,,], obs[matrix_loc,1,]), na.rm = T), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  #add_text(sprintf('RMSE DA: %s \nRMSE no DA: %s', temp_rmse, temp_rmse_no_da), location = 'bottomleft')
  points(obs[matrix_loc,1,] ~ dates, col = 'red', pch = 16, cex = 1.2)
  arrows(dates, obs[matrix_loc,1,]+R[matrix_loc,matrix_loc,], dates, obs[matrix_loc,1,]-R[matrix_loc,matrix_loc,],
         angle = 90, length = .05, col = 'red', code = 3)
  for(i in 1:n_en){
    #lines(Y_no_da[matrix_loc,,i] ~ dates, col = alpha('blue', .5))
    lines(Y[matrix_loc,,i] ~ dates, col = alpha('grey', .5))
  }
  lines(mean_pred ~ dates, lwd = 2, col = alpha('black', .5))
  lines(true[1,,1]~dates, lwd = 2, col = alpha('blue', .5))
  #lines(mean_pred_no_da ~ dates, lwd = 2, col = alpha('blue', .5))
  # abline(v = dates[30])

  #plot(Q[matrix_loc,matrix_loc,] ~ dates, type = 'l',
   #    ylab = 'Process Error', xlab = '', lty=3,lwd = 3,
    #   cex.axis = 2, cex.lab =2)
  true_rmse = round(sqrt(mean((mean_pred - true[1,,1])^2, na.rm = T)), 2)
  obs_rmse = round(sqrt(mean((obs[matrix_loc,1,] - true[1,,1])^2, na.rm = T)), 2)

  windows()
  plot(mean_pred ~ true[1,,1], pch = 16, ylab = 'Predicted or Observed State', xlab = 'True State')
  points(obs[matrix_loc,1,] ~ true[1,,1], pch = 16, col = alpha('red', .5))
  add_text(sprintf('RMSE Preds-True: %s \nRMSE Obs-True: %s \nRMSE Preds-Obs: %s', true_rmse, obs_rmse, temp_rmse), location = 'topleft')
  abline(0,1)
}




# RMSE
# time_period = 1:n_step
# mean_Y = rowMeans(Y, dims = 2) # mean of ensembles for each time step
# mean_temp = mean_Y[1:length(cur_model_idxs), time_period]
# temp_rmse = sqrt(rowMeans((mean_temp - obs[,1,time_period])^2, na.rm = T))
# rmse_all = sqrt(mean((mean_temp - obs[,1,time_period])^2, na.rm = T))
#
# hist(temp_rmse)



# h and c
#lordville site is seg_id_nat == 1573; model_idx = 224
# cur_model_idxs = '1573'
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc_h = which(cur_model_idxs == j) + 1*n_segs
  matrix_loc_c = which(cur_model_idxs == j) + 2*n_segs
  mean_h = rowMeans(Y[matrix_loc_h,,])
  mean_c = rowMeans(Y[matrix_loc_c,,])
  mean_h_no_da = rowMeans(Y_no_da[matrix_loc_h,,])
  mean_c_no_da = rowMeans(Y_no_da[matrix_loc_c,,])

  windows(width = 14, height = 10)
  par(mar = c(3,6,4,3), mfrow = c(2,1))
  plot(Y[matrix_loc_h,,1] ~ dates, type = 'l',
       ylab = 'h', xlab = '', lty=0, ylim =range(c(Y[matrix_loc_h,,], na.rm = T)), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  for(i in 1:n_en){
    lines(Y[matrix_loc_h,,i] ~ dates, col = alpha('grey', .5))
    lines(Y_no_da[matrix_loc_h,,i] ~ dates, col = alpha('blue', .5))
  }
  lines(mean_h ~ dates, lwd = 2, col = alpha('black', .5))
  lines(mean_h_no_da ~ dates, lwd = 2, col = alpha('blue', .9))

  plot(Y[matrix_loc_c,,1] ~ dates, type = 'l',
       ylab = 'c', xlab = '', lty=0, ylim =range(c(Y[matrix_loc_c,,], na.rm = T)), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  for(i in 1:n_en){
    lines(Y[matrix_loc_c,,i] ~ dates, col = alpha('grey', .5))
    lines(Y_no_da[matrix_loc_c,,i] ~ dates, col = alpha('blue', .5))
  }
  lines(mean_c ~ dates, lwd = 2, col = alpha('black', .5))
  lines(mean_c_no_da ~ dates, lwd = 2, col = alpha('blue', .5))

}


# h and cQ
#lordville site is seg_id_nat == 1573; model_idx = 224
# cur_model_idxs = '1573'
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc_h = which(cur_model_idxs == j) + 1*n_segs
  matrix_loc_c = which(cur_model_idxs == j) + 2*n_segs

  windows(width = 14, height = 10)
  par(mar = c(3,6,4,3), mfrow = c(2,1))
  plot(Q[matrix_loc_h,matrix_loc_h,] ~ dates, type = 'l',
       ylab = 'h Process Error', xlab = '', lty=3,lwd = 3,
       cex.axis = 2, cex.lab =2)

  plot(Q[matrix_loc_c,matrix_loc_c,] ~ dates, type = 'l',
       ylab = 'h Process Error', xlab = '', lty=3,lwd = 3,
       cex.axis = 2, cex.lab =2)

}


# h and c ~ LSTM error
#lordville site is seg_id_nat == 1573; model_idx = 224
# cur_model_idxs = '1573'
for(j in cur_model_idxs){
  matrix_loc = which(cur_model_idxs == j)
  matrix_loc_h = which(cur_model_idxs == j) + 1*n_segs
  matrix_loc_c = which(cur_model_idxs == j) + 2*n_segs
  mean_h = rowMeans(Y[matrix_loc_h,,])
  mean_c = rowMeans(Y[matrix_loc_c,,])
  mean_h_no_da = rowMeans(Y_no_da[matrix_loc_h,,])
  mean_c_no_da = rowMeans(Y_no_da[matrix_loc_c,,])

  lstm_error = mean_pred_no_da - obs[matrix_loc,1,]

  h_diff = mean_h - mean_h_no_da
  c_diff = mean_c - mean_c_no_da

  windows(width = 14, height = 10)
  par(mar = c(6,6,4,3), mfrow = c(1,2))
  plot(h_diff ~ lstm_error,
       ylab = 'h adjustment', xlab = 'LSTM error', lty=0, ylim = range(h_diff[!is.na(lstm_error)]),
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  abline(v = 0, h = 0, lty = 2)
  plot(c_diff ~ lstm_error,
       ylab = 'c adjustment', xlab = 'LSTM error', lty=0, ylim = range(c_diff[!is.na(lstm_error)]),
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  abline(v = 0, h = 0, lty = 2)

}


median_P = matrix(nrow = dim(P)[1], ncol = dim(P)[2])
for(i in 1:nrow(median_P)){
  for(j in 1:ncol(median_P)){
    median_P[i,j] = median(P[i,j,])
  }
}

library(plot.matrix)

# P through time
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc = which(cur_model_idxs == j)

  windows(width = 14, height = 10)
  par(mar = c(5,6,4,5), mfrow = c(2,1))
  plot(P[matrix_loc,matrix_loc,] ~ dates, type = 'l',
       ylab = 'Sample Covariance', xlab = '', lty=3,lwd = 3,
       cex.axis = 2, cex.lab =2)

  plot(median_P, main = 'Median Covariance Matrix')
}


# plot forecasts
Y_forecast[,,1,] = Y[1:n_segs,,]
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc = which(cur_model_idxs == j)

  windows(width = 14, height = 10)
  plot(Y_forecast[matrix_loc,,1,1] ~ dates, type = 'l',
       ylab = 'Stream Temp (C)', xlab = '', lty=0,
       ylim = c(0,25), #ylim =range(c(Y[matrix_loc,,], obs[matrix_loc,1,]), na.rm = T), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  points(obs[matrix_loc,1,] ~ dates, col = 'red', pch = 16, cex = 1.2)
  arrows(dates, obs[matrix_loc,1,]+R[matrix_loc,matrix_loc,], dates, obs[matrix_loc,1,]-R[matrix_loc,matrix_loc,],
         angle = 90, length = .05, col = 'red', code = 3)
  for(t in 1:n_step){
    mean_pred = rowMeans(Y_forecast[matrix_loc,t,,])
    cur_dates = dates[t:(t+f_horizon-1)]
    for(i in 1:n_en){
      lines(Y_forecast[matrix_loc,t,,i] ~ cur_dates, col = alpha('grey', .5))
    }
    lines(mean_pred ~ cur_dates, lwd = 2, col = alpha('black', .5))
  }
}

