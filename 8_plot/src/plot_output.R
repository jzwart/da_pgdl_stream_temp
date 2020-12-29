
library(dplyr)
library(ggplot2)
library(reticulate)
np = import('numpy')

d = np$load('5_pgdl_pretrain/out/simple_lstm_da.npz')

obs = d$f[['obs']] #[,,1:10]
Y = d$f[['Y']]#[,1:10,]
R = d$f[['R']]#[,,1:10]
Q = d$f[['Q']]
P = d$f[['P']]
n_en = 30
dates = d$f[['dates']]
n_step = length(dates)
cur_model_idxs = d$f[['model_locations']]
preds_no_da = d$f[['preds_no_da']]

#lordville site is seg_id_nat == 1573; model_idx = 224
# cur_model_idxs = '1573'
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc = which(cur_model_idxs == j)
  mean_pred = rowMeans(Y[matrix_loc,,])
  mean_pred_no_da = colMeans(preds_no_da[,,matrix_loc])

  windows(width = 14, height = 10)
  par(mar = c(3,6,4,3), mfrow = c(2,1))
  plot(Y[matrix_loc,,1] ~ dates, type = 'l',
       ylab = 'Stream Temp (C)', xlab = '', lty=0,
       ylim = c(0,25), #ylim =range(c(Y[matrix_loc,,], obs[matrix_loc,1,]), na.rm = T), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  points(obs[matrix_loc,1,] ~ dates, col = 'red', pch = 16, cex = 1.2)
  arrows(dates, obs[matrix_loc,1,]+R[matrix_loc,matrix_loc,], dates, obs[matrix_loc,1,]-R[matrix_loc,matrix_loc,],
         angle = 90, length = .05, col = 'red', code = 3)
  for(i in 1:n_en){
    # lines(Y_no_assim[matrix_loc,,i] ~ d$dates, col = 'grey')
    lines(Y[matrix_loc,,i] ~ dates, col = alpha('grey', .5))
  }
  lines(mean_pred ~ dates, lwd = 2, col = alpha('black', .5))
  lines(mean_pred_no_da ~ dates, lwd = 2, col = alpha('blue', .5))

  plot(Q[matrix_loc,matrix_loc,] ~ dates, type = 'l',
       ylab = 'Process Error', xlab = '', lty=3,lwd = 3,
       cex.axis = 2, cex.lab =2)
}


# RMSE
time_period = 1:n_step
mean_Y = rowMeans(Y, dims = 2) # mean of ensembles for each time step
mean_temp = mean_Y[1:length(cur_model_idxs), time_period]
temp_rmse = sqrt(rowMeans((mean_temp - obs[,1,time_period])^2, na.rm = T))
rmse_all = sqrt(mean((mean_temp - obs[,1,time_period])^2, na.rm = T))

hist(temp_rmse)

