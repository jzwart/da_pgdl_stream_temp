
#' retreive the model time steps based on start and stop dates and time step
#'
#' @param model_start model start date in date class
#' @param model_stop model stop date in date class
#' @param time_step model time step, defaults to daily timestep
get_model_dates = function(model_start, model_stop, time_step = 'days'){

  model_dates = seq.Date(from = as.Date(model_start), to = as.Date(model_stop), by = time_step)

  return(model_dates)
}

#' vector for holding states and parameters for updating
#'
#' @param n_states number of states we're updating in data assimilation routine
#' @param n_params_est number of parameters we're calibrating
#' @param n_step number of model timesteps
#' @param n_en number of ensembles
get_Y_vector = function(n_states, n_params_est, n_covar_inf_factor, n_step, n_en){

  Y = array(dim = c(n_states + n_params_est + n_covar_inf_factor, n_step, n_en))

  return(Y)
}

#' observation error matrix, should be a square matrix where
#'   col & row = the number of states and params for which you have observations
#'
#' @param n_states number of states we're updating in data assimilation routine
#' @param n_param_obs number of parameters for which we have observations
#' @param n_step number of model timesteps
#' @param state_sd vector of state observation standard deviation; assuming sd is constant through time
#' @param param_sd vector of parmaeter observation standard deviation; assuming sd is constant through time
get_obs_error_matrix = function(n_states, n_params_obs, n_step, state_sd, param_sd){

  R = array(0, dim = c(n_states + n_params_obs, n_states + n_params_obs, n_step))

  state_var = state_sd^2 #variance of temperature observations

  param_var = c()
  if(length(names(param_sd)) > 0){
    for(i in seq_along(names(param_sd))){
      param_var = c(param_var, param_sd[[names(param_sd)[i]]]^2)
    }
  }

  if(n_params_obs > 0){
    all_var = c(state_var, param_var)
  }else{
    all_var = state_var
  }

  for(i in 1:n_step){
    # variance is the same for each depth and time step; could make dynamic or varying by time step if we have good reason to do so
    R[,,i] = diag(all_var, n_states + n_params_obs, n_states + n_params_obs)
  }

  return(R)
}

#' Measurement operator matrix saying 1 if there is observation data available, 0 otherwise
#'
#' @param n_states number of states we're updating in data assimilation routine
#' @param n_param_obs number of parameters for which we have observations
#' @param n_params_est number of parameters we're calibrating
#' @param n_step number of model timesteps
#' @param obs observation matrix created with get_obs_matrix function
get_obs_id_matrix = function(n_states_obs,
                             n_states_est,
                             n_params_obs,
                             n_params_est,
                             n_covar_inf_factor,
                             n_step,
                             obs){

  H = array(0, dim=c(n_states_obs + n_params_obs, n_states_est + n_params_est + n_covar_inf_factor, n_step))

  # order goes 1) states, 2)params for which we have obs, 3) params for which we're estimating but don't have obs 4) covariance inflation factor if estimated

  for(t in 1:n_step){
    H[1:(n_states_obs + n_params_obs), 1:(n_states_obs + n_params_obs), t] = diag(ifelse(is.na(obs[,,t]),0, 1), n_states_obs + n_params_obs, n_states_obs + n_params_obs)
  }

  return(H)
}


#' turn observation dataframe into matrix
#'
#' @param obs_df observation data frame
#' @param model_dates dates over which you're modeling
#' @param model_locations locations where you're estimating temperature
#' @param n_step number of model time steps
#' @param n_states number of states we're updating in data assimilation routine
get_obs_matrix = function(obs_df, model_dates, model_locations, n_step, n_states){

  # need to know location and time of observation
   # model_locations is arranged by model_idx

  obs_df_filtered = obs_df %>%
    dplyr::filter(seg_id_nat %in% model_locations,
                  date %in% model_dates) %>%
    select(seg_id_nat, date, temp_C) %>%
    group_by(seg_id_nat) %>%
    mutate(site_row = which(model_locations %in% seg_id_nat),  # getting which row in Y vector corresponds to site location
           date_step = which(model_dates %in% date)) %>%
    ungroup()

  obs_matrix = array(NA, dim = c(n_states, 1, n_step))

  for(i in 1:length(model_locations)){
    cur_site = dplyr::filter(obs_df_filtered, site_row == i)
    if(nrow(cur_site) > 0){
      for(j in cur_site$date_step){
        obs_matrix[i, 1, j] = dplyr::filter(obs_df_filtered,
                                            site_row == i,
                                            date_step == j) %>%
          pull(temp_C)
      }
    }else{
      next
    }
  }

  return(obs_matrix)
}



##' @param Y vector for holding states and parameters you're estimating
##' @param R observation error matrix
##' @param obs observations at current timestep
##' @param H observation identity matrix
##' @param n_en number of ensembles
##' @param cur_step current model timestep
kalman_filter = function(Y,
                         R,
                         obs,
                         H,
                         n_en,
                         cur_step,
                         covar_inf_factor,
                         n_states_est,
                         n_params_est,
                         n_covar_inf_factor){

  cur_obs = obs[ , , cur_step]

  cur_obs = ifelse(is.na(cur_obs), 0, cur_obs) # setting NA's to zero so there is no 'error' when compared to estimated states

  ###### estimate the spread of your ensembles #####
  Y_mean = matrix(apply(Y[ , cur_step, ], MARGIN = 1, FUN = mean), nrow = length(Y[ , 1, 1])) # calculating the mean of each temp and parameter estimate
  delta_Y = Y[ , cur_step, ] - matrix(rep(Y_mean, n_en), nrow = length(Y[ , 1, 1])) # difference in ensemble state/parameter and mean of all ensemble states/parameters

  if(covar_inf_factor){ # check to see if I have this right
    covar_inf_mean = mean(Y_mean[(n_states_est+n_params_est+1):(n_states_est+n_params_est+n_covar_inf_factor), ]) # just taking mean for now because I don't think it's easy to make this inflation for any one segment

    # estimate Kalman gain #
    K = ((1 / (n_en - 1)) * covar_inf_mean * delta_Y %*% t(delta_Y) %*% t(H[, , cur_step])) %*%
      qr.solve(((1 / (n_en - 1)) * covar_inf_mean * H[, , cur_step] %*% delta_Y %*% t(delta_Y) %*% t(H[, , cur_step]) + R[, , cur_step]))
  }else{
    # estimate Kalman gain w/o covar_inf_factor #
    K = ((1 / (n_en - 1)) * delta_Y %*% t(delta_Y) %*% t(H[, , cur_step])) %*%
      qr.solve(((1 / (n_en - 1)) * H[, , cur_step] %*% delta_Y %*% t(delta_Y) %*% t(H[, , cur_step]) + R[, , cur_step]))
  }

  # update Y vector #
  for(q in 1:n_en){
    Y[, cur_step, q] = Y[, cur_step, q] + K %*% (cur_obs - H[, , cur_step] %*% Y[, cur_step, q]) # adjusting each ensemble using kalman gain and observations
  }
  return(Y)
}



#' initialize Y vector with draws from distribution of obs
#'
#' @param Y Y vector
#' @param obs observation matrix
initialize_Y = function(Y, init_states, init_params, init_covar_inf_factor,
                        n_states_est, n_states_obs, n_params_est,
                        n_params_obs, n_covar_inf_factor, n_step, n_en,
                        state_sd, param_sd, covar_inf_factor_sd){

  # initializing states with end of spinup from SNTemp ic files (or obs if available)
  if(n_states_est > 0){
    state_names = colnames(init_states)[3:ncol(init_states)]
    first_states = c()
    for(i in 1:length(state_names)){
      cur_state = init_states %>%
        arrange(as.numeric(model_idx)) %>% pull(2+i)
      first_states = c(first_states, cur_state)
    }
    first_states = as.numeric(first_states)
  }else{
    first_states = NULL
  }

  param_sd_vec = c()
  if(n_params_est > 0){
    param_names = names(init_params) #colnames(init_params)[3:ncol(init_params)]
    first_params = c()
    for(i in seq_along(param_names)){
      cur_param = init_params[[param_names[i]]] #  %>%
        # arrange(as.numeric(model_idx)) %>% pull(2+i)
      first_params = c(first_params, cur_param)
      param_sd_vec = param_sd[[param_names[i]]]
    }
    first_params = as.numeric(first_params)
    param_sd_vec = as.numeric(param_sd_vec)
  }else{
    first_params = NULL
  }

  if(n_covar_inf_factor > 0){
    first_covar_inf_factors = init_covar_inf_factor
    covar_inf_factor_sd_vec = rep(covar_inf_factor_sd, n_covar_inf_factor)
  }else{
    first_covar_inf_factors = NULL
    covar_inf_factor_sd_vec = NULL
  }

  Y[ , 1, ] = array(rnorm(n = n_en * (n_states_est + n_params_est + n_covar_inf_factor),
                          mean = c(first_states, first_params, first_covar_inf_factors),
                          sd = c(state_sd, param_sd_vec, covar_inf_factor_sd_vec)),
                    dim = c(c(n_states_est + n_params_est + n_covar_inf_factor), n_en))

  return(Y)
}

get_updated_params = function(Y, param_names, n_states_est, n_params_est, cur_step, en, model_run_loc, param_default_file){

  updated_params = Y[(n_states_est+1):(n_states_est+n_params_est), cur_step, en]

  out = vector(mode = 'list', length = length(param_names))

  if(length(param_names) == 0){
    out = out
  }else{
    for(i in seq_along(param_names)){

      defaults = get_default_param_vals(param_name = param_names[i],
                                        model_run_loc = model_run_loc,
                                        param_default_file = param_default_file)

      param_loc_start = as.numeric(defaults$size) * (i-1) + 1
      param_loc_end = param_loc_start + as.numeric(defaults$size) - 1

      cur_param_vals = updated_params[param_loc_start:param_loc_end]

      if(any(as.numeric(cur_param_vals) <= as.numeric(defaults$min))){
        range = as.numeric(defaults$max) - as.numeric(defaults$min)
        # add quarter of range from min
        to_add = range *.05
        cur_param_vals[as.numeric(cur_param_vals) <= as.numeric(defaults$min)] = as.character(as.numeric(cur_param_vals[as.numeric(cur_param_vals) <= as.numeric(defaults$min)]) + to_add)
      }
      if(defaults$type == '1'){
        cur_param_vals = as.character(round(as.numeric(cur_param_vals), digits = 0))
      }

      out[[i]] = cur_param_vals
      names(out)[i] = param_names[i]
    }
  }

  return(out)
}

get_updated_covar_inf_factor = function(Y,
                                        n_states_est,
                                        n_params_est,
                                        n_covar_inf_factor,
                                        cur_step,
                                        en){

  out = Y[(n_states_est+n_params_est+1):(n_states_est+n_params_est+n_covar_inf_factor), cur_step, en]

  return(out)
}


get_updated_states = function(Y, state_names, n_states_est, n_params_est, cur_step, en){

  updated_states = Y[1:n_states_est, cur_step, en]

  return(updated_states)
}

gather_states = function(ic_out){

  state_names = colnames(ic_out)[3:ncol(ic_out)]
  states = c()
  for(i in 1:length(state_names)){
    cur_state = ic_out %>%
      arrange(as.numeric(model_idx)) %>% pull(2+i)
    states = c(states, cur_state)
  }
  states = as.numeric(states)

  return(states)
}


model_spinup = function(n_en,
                        start,
                        stop,
                        time_step = 'days',
                        model_run_loc = '4_model/tmp',
                        spinup_days = 730){

  n_en = as.numeric(n_en)
  spinup_days = as.numeric(spinup_days)
  start = as.Date(as.character(start))
  stop = as.Date(as.character(stop))
  dates = get_model_dates(model_start = start, model_stop = stop, time_step = time_step)

  for(n in 1:n_en){
    run_sntemp(start = dates[1], stop = dates[1], spinup = T,
               model_run_loc = model_run_loc,
               spinup_days = spinup_days,
               restart = T,
               precip_file = sprintf('./input/prcp_%s.cbh', n),
               tmax_file = sprintf('./input/tmax_%s.cbh', n),
               tmin_file = sprintf('./input/tmin_%s.cbh', n),
               var_init_file = sprintf('prms_ic_spinup_%s.txt', n),
               var_save_file = sprintf('prms_ic_spinup_%s.txt', n))
  }
}

add_process_error = function(preds,
                             dates,
                             model_idx,
                             state_error,
                             alpha,
                             beta,
                             R,
                             obs,
                             H,
                             n_en,
                             cur_step){


  if(length(dates) == 1){
    q = NA
    w = rnorm(length(model_idx), 0, 1)
    q = state_error * w

    preds[1:length(model_idx)] = preds[1:length(model_idx)] + q
    preds[1:length(model_idx)] = ifelse(preds[1:length(model_idx)] < 0, 0, preds[1:length(model_idx)])

    return(preds)
  }else{
    q = NA
    for(i in seq_along(model_idx)){
      w = rnorm(length(dates), 0, 1)
      q[1] = state_error * w[1]
      for(z in 2:length(dates)){
        q[z] = alpha * q[z-1] + sqrt(1-alpha^2) * state_error * w[z]
      }
      preds$water_temp[preds$model_idx == model_idx[i]] = preds$water_temp[preds$model_idx == model_idx[i]] + q
      preds$water_temp = ifelse(preds$water_temp < 0, 0,preds$water_temp)
    }

    return(preds)
  }

}

# get ensemble deviations
get_ens_deviate = function(Y,
                           n_en,
                           cur_step){

  Y_mean = matrix(apply(Y[ , cur_step, ], MARGIN = 1, FUN = mean), nrow = length(Y[ , 1, 1])) # calculating the mean of each temp and parameter estimate
  delta_Y = Y[ , cur_step, ] - matrix(rep(Y_mean, n_en), nrow = length(Y[ , 1, 1])) # difference in ensemble state/parameter and mean of all ensemble states/parameters
  return(delta_Y)
}

#get covariance
get_covar = function(deviations,
                     n_en){
  covar = (1 / (n_en - 1)) * deviations %*% t(deviations)
}

# returns gamma from Restatter
get_error_dist = function(H,
                          P,
                          n_en,
                          cur_step,
                          beta){
  # see Rastetter et al Ecological Applications, 20(5), 2010, pp. 1285–1301
  # cur_H = H[,1:42,cur_step]
  # cur_P = P[1:42, 1:42]
  #
  # gamma = 2
  #   t((1-beta) * qr.solve(cur_H%*%cur_P%*%t(cur_H))
  #   H%*%


}


