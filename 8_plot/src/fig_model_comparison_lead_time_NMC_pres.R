
library(dplyr)
library(tidyverse)
library(ggplot2)
library(reticulate)
library(verification) # for CRPS calculation
np = import('numpy')

seg_id = 1573 # Lordville is 1573; 2046 is a relatively low-impacted segment in the Christina watershed
epochs = 50
hidden_units = 10

d_no_ar1_hc = np$load(sprintf('5_pgdl_pretrain/out/lstm_da_segid[%s]_%sepoch_0.5beta_0.9alpha_Truehc_FalseAR1_%sHiddenUnits_FalseMCdropout.npz', seg_id,epochs, hidden_units))
d_ar1_hc = np$load(sprintf('5_pgdl_pretrain/out/lstm_da_segid[%s]_%sepoch_0.5beta_0.9alpha_Truehc_TrueAR1_%sHiddenUnits_FalseMCdropout.npz', seg_id, epochs, hidden_units))
d_no_ar1_no_hc = np$load(sprintf('5_pgdl_pretrain/out/lstm_da_segid[%s]_%sepoch_0.5beta_0.9alpha_Falsehc_FalseAR1_%sHiddenUnits_FalseMCdropout.npz', seg_id, epochs, hidden_units))
d_ar1_no_hc = np$load(sprintf('5_pgdl_pretrain/out/lstm_da_segid[%s]_%sepoch_0.5beta_0.9alpha_Falsehc_TrueAR1_%sHiddenUnits_FalseMCdropout.npz', seg_id, epochs, hidden_units))

# res_data = read.csv('3_observations/in/reservoir_releases_lordville.csv', stringsAsFactors = F) %>%
#   as_tibble() %>%
#   mutate(date = as.Date(date))

n_en = dim(d_ar1_hc$f['Y'])[3]
dates = d_ar1_hc$f[['dates']]
n_step = length(dates)
cur_model_idxs = d_ar1_hc$f[['model_locations']]
n_segs = length(cur_model_idxs)

obs = d_ar1_hc$f[['obs']]
DL_DA_ar1 = d_ar1_hc$f[['Y_forecasts']]
f_horizon = dim(DL_DA_ar1)[3]
DL_ar1 = d_ar1_no_hc$f[['Y_forecasts']]
DL_DA = array(d_no_ar1_hc$f[['Y_forecasts']][,2:(n_step+1),,], dim=c(n_segs,n_step,f_horizon,n_en)) # need to make model lengths the same as the ar1 models
DL = array(d_no_ar1_no_hc$f[['Y_no_da']][,2:(n_step+1),], dim = c(n_segs,n_step,n_en))# will be the same predictions at all lead times since not using forecasted drivers and not updating states

R = d_ar1_hc$f[['R']]


add_text <- function(text, location="topright"){
  legend(location,legend=text, bty ="n", pch=NA)
}

persistence_forecast = DL_DA_ar1
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


# what is the RMSE based on lead time?
issue_times = 1:(n_step-f_horizon)
out = crossing(tibble(issue_date = dates[issue_times]), tibble(lead_time = seq(0,f_horizon-1)))
out = mutate(out,
             valid_time = issue_date + lubridate::days(lead_time),
             # mean_DL_forecast = NA,
             # sd_DL_forecast = NA,
             mean_DL_ar1_forecast = NA,
             sd_DL_ar1_forecast = NA,
             mean_DL_DA_forecast = NA,
             sd_DL_DA_forecast = NA,
             mean_DL_DA_ar1_forecast = NA,
             sd_DL_DA_ar1_forecast = NA,
             persistence = NA)
for(t in issue_times){
  cur_date = dates[t]
  matrix_loc = which(cur_model_idxs == cur_model_idxs)
  # mean_DL_pred = mean(DL[matrix_loc,t+,])
  # sd_DL_pred = sd(DL[matrix_loc,t,])

  mean_DL_ar1_pred = rowMeans(DL_ar1[matrix_loc,t,,])
  sd_DL_ar1_pred = apply(DL_ar1[matrix_loc,t,,], 1, sd)
  mean_DL_DA_pred = rowMeans(DL_DA[matrix_loc,t,,])
  sd_DL_DA_pred = apply(DL_DA[matrix_loc,t,,], 1, sd)
  mean_DL_DA_ar1_pred = rowMeans(DL_DA_ar1[matrix_loc,t,,])
  sd_DL_DA_ar1_pred = apply(DL_DA_ar1[matrix_loc,t,,], 1, sd)

  persistence = rowMeans(persistence_forecast[matrix_loc,t,,])
  # ci = apply(Y_forecast[matrix_loc,t,,], 1, Rmisc::CI)
  out$mean_DL_ar1_forecast = ifelse(out$issue_date==cur_date, mean_DL_ar1_pred, out$mean_DL_ar1_forecast)
  out$sd_DL_ar1_forecast = ifelse(out$issue_date==cur_date, sd_DL_ar1_pred, out$sd_DL_ar1_forecast)
  out$mean_DL_DA_forecast = ifelse(out$issue_date==cur_date, mean_DL_DA_pred, out$mean_DL_DA_forecast)
  out$sd_DL_DA_forecast = ifelse(out$issue_date==cur_date, sd_DL_DA_pred, out$sd_DL_DA_forecast)
  out$mean_DL_DA_ar1_forecast = ifelse(out$issue_date==cur_date, mean_DL_DA_ar1_pred, out$mean_DL_DA_ar1_forecast)
  out$sd_DL_DA_ar1_forecast = ifelse(out$issue_date==cur_date, sd_DL_DA_ar1_pred, out$sd_DL_DA_ar1_forecast)
  out$persistence = ifelse(out$issue_date==cur_date, persistence, out$persistence)
}
obs_df = tibble(date = dates[issue_times], obs_temp = obs[1,1,issue_times])
mean_DL_pred = rowMeans(DL[matrix_loc,,])
sd_DL_pred = apply(DL[matrix_loc,,], 1, sd)
DL_preds = tibble(date = dates[issue_times], mean_DL_forecast = mean_DL_pred[issue_times], sd_DL_forecast = sd_DL_pred[issue_times])
out = left_join(out,obs_df, by = c("valid_time"=  "date"))
out = left_join(out, DL_preds, by = c("valid_time" = "date"))

RMSE = function(m, o, na.rm = T){
  sqrt(mean((m - o)^2, na.rm = na.rm))
}

accuracy_sum = out %>%
  group_by(lead_time) %>%
  summarise(DL_forecast = RMSE(mean_DL_forecast, obs_temp),
            DL_ar1_forecast = RMSE(mean_DL_ar1_forecast, obs_temp),
            DL_DA_forecast = RMSE(mean_DL_DA_forecast, obs_temp),
            DL_DA_ar1_forecast = RMSE(mean_DL_DA_ar1_forecast, obs_temp),
            persistence_forecast = RMSE(persistence, obs_temp)) %>%
  pivot_longer(cols = contains('forecast'), names_to = 'forecast_type',values_to = 'rmse') %>% ungroup() %>%
  mutate(color = case_when(forecast_type == 'persistence_forecast' ~ 'orange',
                           forecast_type == 'DL_forecast' ~ 'purple',
                           forecast_type == 'DL_DA_ar1_forecast' ~ 'darkgreen'))
windows()
persist = ggplot(filter(accuracy_sum, lead_time >0, forecast_type == 'persistence_forecast'),
                 aes(x = lead_time, y = rmse, group = forecast_type, color = forecast_type))+
  geom_line(size = 2) +
  geom_point(size = 3) +
  theme_classic()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 18))+
  xlab('Lead Time (days)') +
  ylab('RMSE (C)') +
  scale_x_continuous(breaks = 1:7)+
  ylim(range(accuracy_sum$rmse[accuracy_sum$lead_time>0])) +
  scale_color_manual(name = "Model Type", values = filter(accuracy_sum, lead_time >0, forecast_type == 'persistence_forecast') %>% pull(color),
                      labels = c("Persistence"))

persist

ggsave('8_plot/out/[1573]_persistence.png', persist, width = 7, height = 7, dpi = 300)

dl_persist = ggplot(filter(accuracy_sum, lead_time >0, forecast_type %in% c('DL_forecast', 'persistence_forecast')),
                 aes(x = lead_time, y = rmse, group = forecast_type, color = forecast_type))+
  geom_line(size = 2) +
  geom_point(size = 3) +
  theme_classic()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 18))+
  xlab('Lead Time (days)') +
  ylab('RMSE (C)') +
  scale_x_continuous(breaks = 1:7)+
  ylim(range(accuracy_sum$rmse[accuracy_sum$lead_time>0])) +
  scale_color_manual(name = "Model Type",
                     values = filter(accuracy_sum, lead_time >0, forecast_type %in% c('DL_forecast', 'persistence_forecast')) %>% pull(color),
                     labels = c("PGDL", "Persistence"))

dl_persist

ggsave('8_plot/out/[1573]_dl_persistence.png', dl_persist, width = 7, height = 7, dpi = 300)


dl_dlda_persist = ggplot(filter(accuracy_sum, lead_time >0, forecast_type %in% c('DL_forecast', 'DL_DA_ar1_forecast','persistence_forecast')),
                    aes(x = lead_time, y = rmse, group = forecast_type, color = forecast_type))+
  geom_line(size = 2) +
  geom_point(size = 3) +
  theme_classic()+
  theme(axis.text = element_text(size =14),
        axis.title = element_text(size = 18))+
  xlab('Lead Time (days)') +
  ylab('RMSE (C)') +
  scale_x_continuous(breaks = 1:7)+
  ylim(range(accuracy_sum$rmse[accuracy_sum$lead_time>0])) +
  scale_color_manual(name = "Model Type",
                     values = filter(accuracy_sum, lead_time >0,
                                     forecast_type %in% c('DL_forecast', 'DL_DA_ar1_forecast','persistence_forecast')) %>% pull(color),
                     labels = c("PGDL + DA", "PGDL","Persistence"))

dl_dlda_persist

ggsave('8_plot/out/[1573]_dl_dlda_persistence.png', dl_dlda_persist, width = 7, height = 7, dpi = 300)

# CRPS calculation
# accuracy_crps = tibble(lead_time = seq(0,f_horizon-1), DA_crps = NA, No_DA_crps = NA)
# for(i in accuracy_crps$lead_time){
#   cur_out = dplyr::filter(dplyr::select(out, lead_time, mean_da_forecast, sd_da_forecast, obs_temp, mean_pred_no_da), lead_time == i)
#   cur_crps = crps(obs = cur_out$obs_temp, pred = as.matrix(cur_out[c('mean_da_forecast','sd_da_forecast')]))
#   accuracy_crps$DA_crps[accuracy_crps$lead_time == i] = mean(cur_crps$crps, na.rm = T)
#   # brier(obs = cur_out$obs_temp, pred = as.matrix(cur_out[c('mean_da_forecast','sd_da_forecast')]))
# }
# windows()
# ggplot(filter(accuracy_crps, lead_time >0), aes(x = lead_time, y = DA_crps))+
#   geom_line(size = 2) +
#   geom_point(size = 3) +
#   theme_minimal()+
#   theme(axis.text = element_text(size =14),
#         axis.title = element_text(size = 16))+
#   xlab('Lead Time (days)') +
#   ylab('CRPS (C)')
#   # xlim(c(1,f_horizon-1))+
#   # scale_color_discrete(name = "Model Type",
  #                      labels = c("DA w/ AR1",'DA w/o AR1', 'AR1 w/o DA', "No DA or AR1", "Persistence"))
