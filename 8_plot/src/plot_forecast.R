
x_range = 17:30

cur_model_idxs = '1573'
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc = which(cur_model_idxs == j)
  mean_pred = rowMeans(Y[matrix_loc,,])
  mean_pred_no_da = rowMeans(Y_no_da[matrix_loc,,]) # colMeans(preds_no_da[,,matrix_loc])

  temp_rmse = round(sqrt(mean((mean_pred - obs[matrix_loc,1,])^2, na.rm = T)), 2)
  temp_rmse_no_da = round(sqrt(mean((mean_pred_no_da - obs[matrix_loc,1,])^2, na.rm = T)), 2)

  windows(width = 14, height = 10)
  par(mar = c(3,6,4,3)) #, mfrow = c(2,1))
  plot(Y[matrix_loc,,1] ~ dates, type = 'l',
       ylab = 'Stream Temp (C)', xlab = '', lty=0,
       xlim = range(dates[x_range]),
       ylim =range(c(Y[matrix_loc,x_range,], obs[matrix_loc,1,x_range]), na.rm = T),
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  add_text(sprintf('RMSE DA: %s \nRMSE no DA: %s', temp_rmse, temp_rmse_no_da), location = 'bottomleft')
  points(obs_withheld[matrix_loc,1,] ~ dates, col = 'purple', pch = 16, cex = 1.2)
  arrows(dates, obs_withheld[matrix_loc,1,]+R[matrix_loc,matrix_loc,], dates, obs_withheld[matrix_loc,1,]-R[matrix_loc,matrix_loc,],
         angle = 90, length = .05, col = 'purple', code = 3)
  points(obs[matrix_loc,1,] ~ dates, col = 'red', pch = 16, cex = 1.2)
  arrows(dates, obs[matrix_loc,1,]+R[matrix_loc,matrix_loc,], dates, obs[matrix_loc,1,]-R[matrix_loc,matrix_loc,],
         angle = 90, length = .05, col = 'red', code = 3)
  for(i in 1:n_en){
    lines(Y_no_da[matrix_loc,,i] ~ dates, col = alpha('blue', .5))
    lines(Y[matrix_loc,,i] ~ dates, col = alpha('grey', .5))
  }
  lines(mean_pred ~ dates, lwd = 2, col = alpha('black', .5))
  lines(mean_pred_no_da ~ dates, lwd = 2, col = alpha('blue', .5))
  # abline(v = dates[30])

  # plot(Q[matrix_loc,matrix_loc,] ~ dates, type = 'l',
  #      ylab = 'Process Error', xlab = '', lty=3,lwd = 3,xlim = range(dates[x_range]),
  #      cex.axis = 2, cex.lab =2)
}
