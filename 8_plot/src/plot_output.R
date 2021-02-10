
library(dplyr)
library(tidyverse)
library(ggplot2)
library(reticulate)
library(verification) # for CRPS calculation
np = import('numpy')

d = np$load('5_pgdl_pretrain/out/lstm_da_150epoch_0.5beta_0.9alpha_Truehc_TrueAR1_6HiddenUnits.npz')

obs = d$f[['obs']] #[,,1:10]
obs_withheld = d$f[['obs_orig']] # if we withhold observations from DA steps
Y = d$f[['Y']]#[,1:10,]
R = d$f[['R']]#[,,1:10]
Q = d$f[['Q']]
P = d$f[['P']]
n_en = dim(Y)[3]
dates = d$f[['dates']]
n_step = length(dates)
cur_model_idxs = d$f[['model_locations']]
n_segs = length(cur_model_idxs)
#preds_no_da = d$f[['preds_no_da']]
Y_no_da = d$f[['Y_no_da']]
Y_forecast = d$f[['Y_forecasts']]
f_horizon = dim(Y_forecast)[3]
#true = d$f[['true']]
trn_preds = d$f[['trn_preds']]
obs_trn = d$f[['obs_trn']]
dates_trn = d$f[['trn_dates']]

add_text <- function(text, location="topright"){
  legend(location,legend=text, bty ="n", pch=NA)
}

#lordville site is seg_id_nat == 1573; model_idx = 224; 1574 & 1575 are directly upstream of 1573 and 1577 is directly downstream of 1573
# cur_model_idxs = '1573'
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc = which(cur_model_idxs == j)
  mean_pred = rowMeans(Y[matrix_loc,,])
  mean_pred_no_da = rowMeans(Y_no_da[matrix_loc,,]) # colMeans(preds_no_da[,,matrix_loc])

  temp_rmse = round(sqrt(mean((mean_pred - obs[matrix_loc,1,])^2, na.rm = T)), 2)
  temp_rmse_no_da = round(sqrt(mean((mean_pred_no_da - obs[matrix_loc,1,])^2, na.rm = T)), 2)

  windows(width = 14, height = 10)
  par(mar = c(3,6,4,3), mfrow = c(2,1))
  plot(Y[matrix_loc,,1] ~ dates, type = 'l',
       ylab = 'Stream Temp (C)', xlab = '', lty=0,
       ylim = c(0,25), #ylim =range(c(Y[matrix_loc,,], obs[matrix_loc,1,]), na.rm = T), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  add_text(sprintf('RMSE DA: %s \nRMSE no DA: %s', temp_rmse, temp_rmse_no_da), location = 'bottomleft')

  for(i in 1:n_en){
    lines(Y_no_da[matrix_loc,,i] ~ dates, col = alpha('blue', .5))
    lines(Y[matrix_loc,,i] ~ dates, col = alpha('grey', .5))
  }
  lines(mean_pred ~ dates, lwd = 2, col = alpha('black', .5))
  lines(mean_pred_no_da ~ dates, lwd = 2, col = alpha('blue', .5))
  # abline(v = dates[30])
  points(obs[matrix_loc,1,] ~ dates, col = 'red', pch = 16, cex = 1.2)
  arrows(dates, obs[matrix_loc,1,]+R[matrix_loc,matrix_loc,], dates, obs[matrix_loc,1,]-R[matrix_loc,matrix_loc,],
         angle = 90, length = .05, col = 'red', code = 3)

  plot(Q[matrix_loc,matrix_loc,] ~ dates, type = 'l',
       ylab = 'Process Error', xlab = '', lty=3,lwd = 3,
       cex.axis = 2, cex.lab =2)
  # true_rmse = round(sqrt(mean((mean_pred - true[1,,1])^2, na.rm = T)), 2)
  # obs_rmse = round(sqrt(mean((obs[matrix_loc,1,] - true[1,,1])^2, na.rm = T)), 2)
  #
  # windows()
  # plot(mean_pred ~ true[1,,1], pch = 16, ylab = 'Predicted or Observed State', xlab = 'True State')
  # points(obs[matrix_loc,1,] ~ true[1,,1], pch = 16, col = alpha('red', .5))
  # add_text(sprintf('RMSE Preds-True: %s \nRMSE Obs-True: %s \nRMSE Preds-Obs: %s', true_rmse, obs_rmse, temp_rmse), location = 'topleft')
  # abline(0,1)
}


persistence_forecast = Y_forecast
for(i in 1:n_step){
  for(j in cur_model_idxs){
    matrix_loc = which(cur_model_idxs == j)
    if(!is.na(obs[matrix_loc,1,i])){ # if there are observations, set persistence to obs
      persistence_forecast[matrix_loc,i,,] = obs[matrix_loc,1,i]
    }else{
      persistence_forecast[matrix_loc,i,,] = NA
    }
  }
}

# plot forecasts
#Y_forecast[,,1,] = Y[1:n_segs,,]
issue_times = 130:140
for(j in cur_model_idxs){
  # obs[,1,1]
  matrix_loc = which(cur_model_idxs == j)

  mean_pred_no_da = rowMeans(Y_no_da[matrix_loc,,]) # colMeans(preds_no_da[,,matrix_loc])
  mean_pred_da = rowMeans(Y[matrix_loc,,]) # colMeans(preds_no_da[,,matrix_loc])

  windows(width = 14, height = 10)
  par(mar = c(6,6,4,3))
  plot(Y_forecast[matrix_loc,issue_times,1,1] ~ dates[issue_times], type = 'l',
       ylab = 'Stream Temp (C)', xlab = '', lty=0,
       #ylim = c(0,25),
       ylim =range(c(Y_forecast[matrix_loc,issue_times,,], obs[matrix_loc,1,issue_times]), na.rm = T), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  points(obs[matrix_loc,1,issue_times] ~ dates[issue_times], col = 'red', pch = 16, cex = 1.2)
  arrows(dates[issue_times], obs[matrix_loc,1,issue_times]+R[matrix_loc,matrix_loc,issue_times], dates[issue_times], obs[matrix_loc,1,issue_times]-R[matrix_loc,matrix_loc,issue_times],
         angle = 90, length = .05, col = 'red', code = 3)
  for(t in issue_times){
    mean_pred = rowMeans(Y_forecast[matrix_loc,t,,])
    cur_dates = dates[t:(t+f_horizon-1)]
    for(i in 1:n_en){
      lines(Y_forecast[matrix_loc,t,,i] ~ cur_dates, col = alpha('grey', .5))
    }
    lines(mean_pred ~ cur_dates, lwd = 2, col = alpha('black', .5))
    # lines(persistence_forecast[matrix_loc,t,,1] ~ cur_dates, lwd = 2, col = alpha('red',.5)) # persistence forecast
  }
  points(obs[matrix_loc,1,issue_times] ~ dates[issue_times], col = 'red', pch = 16, cex = 1.2)
  arrows(dates[issue_times], obs[matrix_loc,1,issue_times]+R[matrix_loc,matrix_loc,issue_times], dates[issue_times], obs[matrix_loc,1,issue_times]-R[matrix_loc,matrix_loc,issue_times],
         angle = 90, length = .05, col = 'red', code = 3)
  lines(mean_pred_no_da[issue_times] ~ dates[issue_times], lwd =5 ,col = alpha('blue',.5))
  # lines(mean_pred_da[issue_times] ~ dates[issue_times], lwd =2 , lty = 2, col = alpha('green',.9))
}


# what is the RMSE based on lead time?
issue_times = 1:n_step
out = crossing(tibble(issue_date = dates[issue_times]), tibble(lead_time = seq(0,f_horizon-1)))
out = mutate(out,
             valid_time = issue_date + lubridate::days(lead_time),
             mean_da_forecast = NA,
             sd_da_forecast = NA,
             persistence = NA)
for(t in issue_times){
  cur_date = dates[t]
  matrix_loc = which(cur_model_idxs == 1573)
  mean_pred = rowMeans(Y_forecast[matrix_loc,t,,])
  sd_pred = apply(Y_forecast[matrix_loc,t,,], 1, sd)
  persistence = rowMeans(persistence_forecast[matrix_loc,t,,])
  out$mean_da_forecast = ifelse(out$issue_date==cur_date, mean_pred, out$mean_da_forecast)
  out$sd_da_forecast = ifelse(out$issue_date==cur_date, sd_pred, out$sd_da_forecast)
  out$persistence = ifelse(out$issue_date==cur_date, persistence, out$persistence)
}
obs_df = tibble(date = dates[issue_times], obs_temp = obs[1,1,issue_times])
mean_pred_no_da = rowMeans(Y_no_da[matrix_loc,,])
no_da_preds = tibble(date = dates[issue_times], mean_pred_no_da = mean_pred_no_da[issue_times])
out = left_join(out,obs_df, by = c("valid_time"=  "date"))
out = left_join(out, no_da_preds, by = c("valid_time" = "date"))

RMSE = function(m, o, na.rm = T){
  sqrt(mean((m - o)^2, na.rm = na.rm))
}

accuracy_sum = out %>%
  group_by(lead_time) %>%
  summarise(DA_ar1 = RMSE(mean_da_forecast_ar1, obs_temp),
            DA_no_ar1 = RMSE(mean_da_forecast_no_ar1, obs_temp),
            No_DA_ar1 = RMSE(mean_pred_no_da_ar1, obs_temp),
            No_DA_no_ar1 = RMSE(mean_pred_no_da_no_ar1, obs_temp),
            No_DA_persistence = RMSE(persistence, obs_temp)) %>%
  pivot_longer(cols = contains('DA'), names_to = 'forecast_type',values_to = 'rmse')
windows()
ggplot(filter(accuracy_sum, lead_time >0), aes(x = lead_time, y = rmse, group = forecast_type, color = forecast_type))+
  geom_line(size = 2) +
  geom_point(size = 3) +
  theme_minimal()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('Lead Time (days)') +
  ylab('RMSE (C)') +
  # xlim(c(1,f_horizon-1))+
  scale_color_discrete(name = "Model Type",
                      labels = c("DA w/ AR1",'DA w/o AR1', 'AR1 w/o DA', "No DA or AR1", "Persistence"))


# CRPS calculation
accuracy_crps = tibble(lead_time = seq(0,f_horizon-1), DA_crps = NA)
for(i in accuracy_crps$lead_time){
  cur_out = dplyr::filter(dplyr::select(out, lead_time, mean_da_forecast, sd_da_forecast, obs_temp), lead_time == i)
  cur_crps = crps(obs = cur_out$obs_temp, pred = as.matrix(cur_out[c('mean_da_forecast','sd_da_forecast')]))
  accuracy_crps$DA_crps[accuracy_crps$lead_time == i] = mean(cur_crps$crps, na.rm = T)
  # brier(obs = cur_out$obs_temp, pred = as.matrix(cur_out[c('mean_da_forecast','sd_da_forecast')]))
}
windows()
ggplot(filter(accuracy_crps, lead_time >0), aes(x = lead_time, y = DA_crps))+
  geom_line(size = 2) +
  geom_point(size = 3) +
  theme_minimal()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('Lead Time (days)') +
  ylab('CRPS (C)')
  # xlim(c(1,f_horizon-1))+
  # scale_color_discrete(name = "Model Type",
  #                      labels = c("DA w/ AR1",'DA w/o AR1', 'AR1 w/o DA', "No DA or AR1", "Persistence"))



#lordville site is seg_id_nat == 1573; model_idx = 224; 1574 & 1575 are directly upstream of 1573 and 1577 is directly downstream of 1573
# cur_model_idxs = '1573'
for(j in cur_model_idxs){
  # obs[,1,1]
  mean_pred = trn_preds[1,,1]
  cur_obs = obs_trn[1,,1]

  #test period
  matrix_loc = which(cur_model_idxs == j)
  mean_test = rowMeans(Y_no_da[matrix_loc,,])

  trn_rmse = round(sqrt(mean((mean_pred - cur_obs)^2, na.rm = T)), 2)
  test_rmse = round(sqrt(mean((mean_test - obs[matrix_loc,1,])^2, na.rm = T)), 2)

  all_preds = c(mean_pred, mean_test)
  all_dates = c(dates_trn, dates)

  windows(width = 14, height = 10)
  par(mar = c(5,6,4,3), mfrow = c(2,1))
  plot(all_preds ~ all_dates, type = 'l',
       ylab = 'Stream Temp (C)', xlab = '', lty=0,
       ylim = c(0,25), #ylim =range(c(Y[matrix_loc,,], obs[matrix_loc,1,]), na.rm = T), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  points(cur_obs ~ dates_trn, col = 'red', pch = 16, cex = 1.2)
  points(obs[matrix_loc,1,] ~ dates, col = 'red', pch = 16, cex = 1.2)
  lines(mean_pred ~ dates_trn, lwd = 2)
  lines(mean_test ~ dates, lwd = 2, col = 'blue')

  plot(mean_pred~cur_obs, xlab= 'Water Temp Obs (C)', ylab = 'Water Temp Preds (C)',
       cex.axis = 2, cex.lab =2)
  points(mean_test ~ obs[matrix_loc,1,], col = 'blue')
  abline(0,1,col='red')
  add_text(sprintf('RMSE Trn: %s\nRMSE Test: %s', temp_rmse,test_rmse), location = 'topleft')
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
       ylab = 'h', xlab = '', lty=0, ylim =range(c(Y[matrix_loc_h,,]), na.rm = T), #, Y_no_assim[matrix_loc,,])
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  for(i in 1:n_en){
    lines(Y[matrix_loc_h,,i] ~ dates, col = alpha('grey', .5))
    lines(Y_no_da[matrix_loc_h,,i] ~ dates, col = alpha('blue', .5))
  }
  lines(mean_h ~ dates, lwd = 2, col = alpha('black', .5))
  lines(mean_h_no_da ~ dates, lwd = 2, col = alpha('blue', .9))

  plot(Y[matrix_loc_c,,1] ~ dates, type = 'l',
       ylab = 'c', xlab = '', lty=0, ylim =range(c(Y[matrix_loc_c,,]), na.rm = T), #, Y_no_assim[matrix_loc,,])
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
       ylab = 'h adjustment', xlab = 'LSTM error', lty=0, ylim = range(h_diff[!is.na(lstm_error)], na.rm = T),
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  abline(v = 0, h = 0, lty = 2)
  plot(c_diff ~ lstm_error,
       ylab = 'c adjustment', xlab = 'LSTM error', lty=0, ylim = range(c_diff[!is.na(lstm_error)], na.rm = T),
       cex.axis = 2, cex.lab =2, main = sprintf('model idx %s', j))
  abline(v = 0, h = 0, lty = 2)

}


median_P = matrix(nrow = dim(P)[1], ncol = dim(P)[2])
for(i in 1:nrow(median_P)){
  for(j in 1:ncol(median_P)){
    median_P[i,j] = median(P[i,j,], na.rm = T)
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


# what is the RMSE based on lead time?
issue_times = 1:n_step
out = crossing(tibble(issue_date = dates[issue_times]), tibble(lead_time = seq(0,f_horizon-1)))
out = mutate(out,
             valid_time = issue_date + lubridate::days(lead_time),
             mean_da_forecast = NA,
             persistence = NA)
for(t in issue_times){
  cur_date = dates[t]
  mean_pred = rowMeans(Y_forecast[matrix_loc,t,,])
  persistence = rowMeans(persistence_forecast[matrix_loc,t,,])
  out$mean_da_forecast = ifelse(out$issue_date==cur_date, mean_pred, out$mean_da_forecast)
  out$persistence = ifelse(out$issue_date==cur_date, persistence, out$persistence)
}
obs_df = tibble(date = dates[issue_times], obs_temp = obs[1,1,issue_times])
mean_pred_no_da = rowMeans(Y_no_da[matrix_loc,,])
no_da_preds = tibble(date = dates[issue_times], mean_pred_no_da = mean_pred_no_da[issue_times])
out = left_join(out,obs_df, by = c("valid_time"=  "date"))
out = left_join(out, no_da_preds, by = c("valid_time" = "date"))

RMSE = function(m, o, na.rm = T){
  sqrt(mean((m - o)^2, na.rm = na.rm))
}

accuracy_sum = out %>%
  group_by(lead_time) %>%
  summarise(DA = RMSE(mean_da_forecast, obs_temp),
            #No_DA = RMSE(mean_pred_no_da, obs_temp),
            No_DA_persistence = RMSE(persistence, obs_temp)) %>%
  pivot_longer(cols = contains('DA'), names_to = 'forecast_type',values_to = 'rmse')
windows()
ggplot(accuracy_sum, aes(x = lead_time, y = rmse, group = forecast_type, color = forecast_type))+
  geom_line(size = 2) +
  geom_point(size = 3) +
  theme_minimal()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('Lead Time (days)') +
  ylab('RMSE (C)')

#plot((accuracy$rmse_no_da-accuracy$rmse_da) ~ accuracy$lead_time, type = 'l', lwd=3,
#    ylim = c(0, max(accuracy$rmse_no_da-accuracy$rmse_da)))
#abline(0,0,lty=2, lwd=2)

accuracy = out %>%
  #group_by(issue_date) %>%
  #mutate(No_DA_rmse = RMSE(mean_pred_no_da, obs_temp)) %>% ungroup() %>%
  group_by(issue_date, lead_time) %>%
  mutate(DA = abs(mean_da_forecast- obs_temp),
         No_DA_rmse = abs(mean_pred_no_da- obs_temp)) %>% ungroup() %>%
  mutate(rmse_improve = (No_DA_rmse - DA))

windows()
ggplot(accuracy, aes(x = issue_date, y = rmse_improve, color = lead_time, group=  lead_time))+
  geom_point(size = 0) +
  geom_line(size =1) +
  theme_minimal()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('Date') +
  ylab('No DA error - DA error (C)')+
  geom_abline(slope = 0, intercept = 0, linetype = 'dashed', size = 2)

ggplot(accuracy, aes(x = No_DA_rmse, y = rmse_improve, color = lead_time))+
  geom_point(size = 3) +
  theme_minimal()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('LSTM Absolute Error (C)') +
  ylab('No DA error - DA error (C)') +
  geom_abline(slope = 0, intercept = 0, linetype = 'dashed', size = 2)+
  geom_abline(slope = 1, intercept = 0)

ggplot(dplyr::filter(accuracy,lead_time %in% c(1,2,3,4), No_DA_rmse <3), aes(x = No_DA_rmse, y = rmse_improve, color = lead_time, group = lead_time))+
  geom_point(size = 3, alpha = .4) +
  theme_minimal()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('LSTM RMSE (C)') +
  ylab('No DA error - DA error (C)') +
  geom_abline(slope = 0, intercept = 0, linetype = 'dashed', size = 2)+
  geom_abline(slope = 1, intercept = 0) +
  geom_smooth(method = 'lm', se = F, size =2)

ggplot(dplyr::filter(accuracy,lead_time == 1), aes(x = issue_date, y = DA))+
  geom_line() +
  theme_minimal() +
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('Date') +
  ylab('DA error (C)')+
  geom_abline(slope = 0, intercept = mean(accuracy$DA[accuracy$lead_time ==1],na.rm = T),
              linetype ='dashed')

ggplot(dplyr::filter(accuracy,lead_time %in% c(1,2,3,4)), aes(x = issue_date, y = rmse_improve, color = lead_time, group = lead_time))+
  geom_point(size = 1) +
  theme_minimal()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('Date') +
  ylab('No DA error - DA error (C)') +
  geom_abline(slope = 0, intercept = 0, linetype = 'dashed', size = 2)+
  geom_abline(slope = 1, intercept = 0) + geom_smooth(se = F, size =2)

ggplot(dplyr::filter(accuracy,lead_time %in% c(0,1,2,3)), aes(x = valid_time, y = mean_da_forecast, color = lead_time, group = lead_time))+
  geom_line() +
  geom_point(data = dplyr::filter(accuracy,lead_time %in% c(0,1,2,3)), aes(x = valid_time, y = obs_temp))+
  geom_line(data = dplyr::filter(accuracy, lead_time %in% c(0,1,2,3)), aes(x = valid_time, y = mean_pred_no_da), color = 'red') +
  theme_minimal() +
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 16))+
  xlab('Date') +
  ylab('Water temperature (C)')

