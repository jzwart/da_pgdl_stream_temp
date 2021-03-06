import numpy as np
import xarray as xr
#import netCDF4 as nc 
#import datetime as dt
import pandas as pd

def check_if_finite(arr):
    assert np.isfinite(arr.to_array().values).all()

def byte_to_str(arr):
    out = ["" for x in range(arr.shape[0])]
    for i in range(arr.shape[0]):
        out[i] = arr[i].decode('UTF-8') 
    out = np.array(out)
    return out

def scale(data_arr, std=None, mean=None):
    """
    scale the data so it has a standard deviation of 1 and a mean of zero
    :param data_arr: [xarray dataset] input or output data with dims
    [nseg, ndates, nfeats]
    :param std: [numpy array] standard deviation if scaling test data with dims
    [nfeats]
    :param mean: [numpy array] mean if scaling test data with dims [nfeats]
    :return: scaled data with original dims
    """
    if not isinstance(std, xr.Dataset) or not isinstance(mean, xr.Dataset):
        std = data_arr.std(skipna=True)
        mean = data_arr.mean(skipna=True)
    # adding small number in case there is a std of zero
    scaled = (data_arr - mean) / (std + 1e-10)
    check_if_finite(std)
    check_if_finite(mean)
    return scaled, std, mean

def get_nc_data(
    var_file,
    vars,
    seg_id,
    start_date_trn,
    end_date_trn,
    start_date_pred,
    end_date_pred,
    scale_data=False,
    ar1_temp=False,
):
    data = xr.open_dataset(var_file)  # open netcdf file with xarray 
    # make sure the indices are in the same order
    data = data.transpose("valid_time", "seg_id_nat", "ens")
    
    dates = data['valid_times']
    dates = dates.values
    dates = byte_to_str(dates)
    trn_date_idxs = np.array(np.where(np.isin(dates, [start_date_trn,end_date_trn]))).reshape(2).tolist()
    pred_date_idxs = np.array(np.where(np.isin(dates, [start_date_pred,end_date_pred]))).reshape(2).tolist()
    timedelta = pd.timedelta_range(start='0 days', end = '12417 days') # need to make this dynamic based on netcdf file
    trn_date_idxs = timedelta[trn_date_idxs]
    pred_date_idxs = timedelta[pred_date_idxs]

    # get just the variable we want
    data = data[vars]
    
    # get just the seg_id that we want
    data = data.loc[dict(seg_id_nat=seg_id)]

    # separate into train/prediction periods
    data_trn = data.loc[dict(valid_time = slice(trn_date_idxs[0], trn_date_idxs[1]))]
    data_pred = data.loc[dict(valid_time = slice(pred_date_idxs[0], pred_date_idxs[1]))]

    # scale data (only used for x_data)
    if scale_data:
        data_trn, mean, std = scale(data_trn)
        if np.isin('seg_tave_water', vars): 
            seg_tave_water_mean = np.array(mean['seg_tave_water'])
            seg_tave_water_std = np.array(std['seg_tave_water'])
        data_pred, _, _ = scale(data_pred, mean, std)
    
    data.close()
    if scale_data and ar1_temp:
        return data_trn, data_pred, seg_tave_water_mean, seg_tave_water_std
    else: 
        return data_trn, data_pred 

def prep_lstm_da(
    var_file,
    vars,
    seg_id,
    start_date_trn,
    end_date_trn,
    start_date_pred,
    end_date_pred,
    scale_data=False,
    ar1_temp=False,
):
    data = xr.open_zarr(var_file)
    # make sure the indices are in the same order
    data = data.transpose("date", "seg_id_nat")

    # get just the variable we want
    data = data[vars]

    # get just the seg_id that we want
    data = data.loc[dict(seg_id_nat=seg_id)]

    # separate into train/prediction periods
    data_trn = data.loc[dict(date=slice(start_date_trn, end_date_trn))]
    data_pred = data.loc[dict(date=slice(start_date_pred, end_date_pred))]

    # scale data (only used for x_data)
    if scale_data:
        data_trn, mean, std = scale(data_trn)
        data_pred, _, _ = scale(data_pred, mean, std)
        if ar1_temp:
            if 'seg_tave_water' in vars:
                mean = np.array(mean['seg_tave_water'])
                std = np.array(std['seg_tave_water'])
            elif 'temp_c' in vars:
                std = np.array(std['temp_c'])
                mean = np.array(mean['temp_c'])
    else:
        if ar1_temp:
            if 'seg_tave_water' in vars:
                data_dont_use, mean, std = scale(data_trn)
                mean = np.array(mean['seg_tave_water'])
                std = np.array(std['seg_tave_water'])
            elif 'temp_c' in vars:
                data_dont_use, mean, std = scale(data_trn)
                std = np.array(std['temp_c'])
                mean = np.array(mean['temp_c'])
    if ar1_temp:
        return data_trn, data_pred, mean, std
    else: 
        return data_trn, data_pred


def fmt_dataset(dataset):
    return dataset.to_array().values


def prep_data_lstm_da(
    obs_temp_file,
    driver_file,
    dist_mat_file, 
    dist_mat_direction, 
    seg_id,
    start_date_trn,
    end_date_trn,
    start_date_pred,
    end_date_pred,
    x_vars=["seg_tave_air"],
    y_vars=["seg_tave_water"],
    obs_vars = ["temp_c"], 
    out_file=None,
    n_en = 1, # number of ensembles for DA - creating n_en batches 
    ar1_temp = False, # include ar1 term or not 
):
    if ar1_temp: 
        x_trn, x_pred, seg_tave_water_mean, seg_tave_water_std = prep_lstm_da(
            driver_file,
            x_vars,
            seg_id,
            start_date_trn,
            end_date_trn,
            start_date_pred,
            end_date_pred,
            scale_data=True,
            ar1_temp = ar1_temp,
        )
    else: 
        x_trn, x_pred = prep_lstm_da(
            driver_file,
            x_vars,
            seg_id,
            start_date_trn,
            end_date_trn,
            start_date_pred,
            end_date_pred,
            scale_data=True,
            ar1_temp = ar1_temp,
        )
    y_trn, y_pred = prep_lstm_da(
        driver_file,
        y_vars,
        seg_id,
        start_date_trn,
        end_date_trn,
        start_date_pred,
        end_date_pred,
        scale_data=False,
    )
    if ar1_temp: 
        obs_trn, obs_pred, obs_mean, obs_std = prep_lstm_da(
            obs_temp_file,
            obs_vars,
            seg_id,
            start_date_trn,
            end_date_trn,
            start_date_pred,
            end_date_pred,
            scale_data=False,
            ar1_temp = ar1_temp,
        )
    else: 
        obs_trn, obs_pred = prep_lstm_da(
            obs_temp_file,
            obs_vars,
            seg_id,
            start_date_trn,
            end_date_trn,
            start_date_pred,
            end_date_pred,
            scale_data=False,
            ar1_temp = ar1_temp,
        )
    
    dates_trn = obs_trn.date.values
    dates_pred = obs_pred.date.values
    
    doy_trn = np.array(pd.DatetimeIndex(dates_trn).dayofyear)
    doy_std = np.std(doy_trn)
    doy_mean = np.mean(doy_trn)
    doy_trn = (doy_trn - doy_mean) / (doy_std + 1e-10)
    doy_pred = np.array(pd.DatetimeIndex(dates_pred).dayofyear)
    doy_pred = (doy_pred - doy_mean) / (doy_std + 1e-10)
    
    # creating n_en training 
    x_trn = fmt_dataset(x_trn) # current shape is [nfeats, seg_len, nseg, n_en]
    x_trn = np.moveaxis(x_trn, 0, -1)
    x_trn = np.moveaxis(x_trn, 1, 0)  # should now be in shape of [nseg, seq_len, n_en, nfeats]
    # we need to make the first axis repeated by n_en 
    # x_trn_out = np.empty((len(seg_id)*n_en, x_trn.shape[1], x_trn.shape[2])) # shape of [nseg*n_en, seq_len, nfeats]
    # the following should ensure that the first axis is sorted by seg_id_nat 
    # for i in range(n_en):
    #     for j in range(len(seg_id)):
    #         cur_idx = i+j
    #         x_trn_out[cur_idx,:,:] = x_trn[j,:,i,:]
    x_trn = np.repeat(x_trn, n_en, axis = 0)  
    # adding noise to predictors 
    #for i in range(x_trn.shape[0]):
        # should make this adjusted not by the scaled drivers 
     #   x_trn[i,:,:] = x_trn[i,:,:] + np.random.normal(scale = 0.2, size = x_trn.shape[1]).reshape((x_trn.shape[1],1))
    
    y_trn = fmt_dataset(y_trn) # current shape is [nfeats, seg_len, nseg, n_en]
    y_trn = np.moveaxis(y_trn, 0, -1)
    y_trn = np.moveaxis(y_trn, 1, 0)  # should now be in shape of [nseg, seq_len, n_en, nfeats]
    # we need to make the first axis repeated by n_en 
    # y_trn_out = np.empty((len(seg_id)*n_en, y_trn.shape[1], y_trn.shape[3])) # shape of [nseg*n_en, seq_len, nfeats]
    # the following should ensure that the first axis is sorted by seg_id_nat 
    # for i in range(n_en):
    #     for j in range(len(seg_id)):
    #         cur_idx = i+j
    #         y_trn_out[cur_idx,:,:] = y_trn[j,:,i,:]
    y_trn = np.repeat(y_trn, n_en, axis = 0)  
    
    obs_trn = fmt_dataset(obs_trn)
    obs_trn = np.moveaxis(obs_trn, 0, -1)
    obs_trn = np.moveaxis(obs_trn, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    obs_trn = np.repeat(obs_trn, n_en, axis = 0)  
    
    x_pred = fmt_dataset(x_pred) # current shape is [nfeats, seg_len, nseg, n_en]
    x_pred = np.moveaxis(x_pred, 0, -1)
    x_pred = np.moveaxis(x_pred, 1, 0)  # should now be in shape of [nseg, seq_len, n_en, nfeats]
    # we need to make the first axis repeated by n_en 
    # x_pred_out = np.empty((len(seg_id)*n_en, x_pred.shape[1], x_pred.shape[3])) # shape of [nseg*n_en, seq_len, nfeats]
    # the following should ensure that the first axis is sorted by seg_id_nat 
    # for i in range(n_en):
    #     for j in range(len(seg_id)):
    #         cur_idx = i+j
    #         x_pred_out[cur_idx,:,:] = x_pred[j,:,i,:]
    x_pred = np.repeat(x_pred, n_en, axis = 0)  
            
    y_pred = fmt_dataset(y_pred) # current shape is [nfeats, seg_len, nseg, n_en]
    y_pred = np.moveaxis(y_pred, 0, -1)
    y_pred = np.moveaxis(y_pred, 1, 0)  # should now be in shape of [nseg, seq_len, n_en, nfeats]
    # we need to make the first axis repeated by n_en 
    # y_pred_out = np.empty((len(seg_id)*n_en, y_pred.shape[1], y_pred.shape[3])) # shape of [nseg*n_en, seq_len, nfeats]
    # the following should ensure that the first axis is sorted by seg_id_nat 
    # for i in range(n_en):
    #     for j in range(len(seg_id)):
    #         cur_idx = i+j
    #         y_pred_out[cur_idx,:,:] = y_pred[j,:,i,:]
    y_pred = np.repeat(y_pred, n_en, axis = 0)  
            
    obs_pred = fmt_dataset(obs_pred)
    obs_pred = np.moveaxis(obs_pred, 0, -1)
    obs_pred = np.moveaxis(obs_pred, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    # obs_pred = np.repeat(obs_pred, n_en, axis = 0)  
    
    # read in distance matrix 
    n_segs = len(seg_id)
    dist_mat_out = np.array(np.ones((n_en*n_segs,n_en*n_segs)) * np.inf)
    rowcolnames_out = np.array(np.ones((n_en*n_segs)) * np.inf)
    dist_mat = np.load(dist_mat_file)
    rowcolnames = dist_mat['rowcolnames'] # seg_id_nat 
    dist_mat = dist_mat[dist_mat_direction] # pull out only direction we want 
    seg_id_char = np.array([str(x) for x in seg_id])
    seg_idxs = np.where(np.isin(rowcolnames, seg_id_char))
    seg_idxs_list = np.array(seg_idxs).reshape((len(seg_id))).tolist()
    # pull out dist matrix for current seg_id_nat 
    dist_mat = dist_mat[seg_idxs_list,:]
    dist_mat = dist_mat[:,seg_idxs_list] 
    for n in range(n_en):
        cur_start = n * n_segs
        cur_end = cur_start + n_segs
        h_fill = np.tile(dist_mat, n+1) # horizontal fill 
        v_fill = np.tile(dist_mat, ((n+1), 1)) # vertical fill 
        dist_mat_out[cur_start:cur_end, 0:cur_end] = h_fill 
        dist_mat_out[0:cur_end, cur_start:cur_end] = v_fill
        rowcolnames_out[cur_start:cur_end] = rowcolnames[seg_idxs_list].astype(int).reshape((len(seg_id)))
    
    dist_mat_out = prep_adj_matrix(mat = dist_mat_out, rowcolnames= rowcolnames_out)
    
    
    if ar1_temp:
        data = {
            "x_trn": x_trn,
            "x_pred": x_pred,
            "dates_trn": dates_trn,
            "dates_pred": dates_pred,
            "model_locations": seg_id, 
            "y_trn": y_trn,
            "y_pred": y_pred,
            "obs_trn": obs_trn,
            "obs_pred": obs_pred,
            "seg_tave_water_mean": seg_tave_water_mean,
            "seg_tave_water_std": seg_tave_water_std,
            "obs_mean": obs_mean,
            "obs_std": obs_std, 
            "doy_trn": doy_trn,
            "doy_pred": doy_pred,
            "distance_matrix": dist_mat_out,
        }
    else: 
        data = {
            "x_trn": x_trn,
            "x_pred": x_pred,
            "dates_trn": dates_trn,
            "dates_pred": dates_pred,
            "model_locations": seg_id, 
            "y_trn": y_trn,
            "y_pred": y_pred,
            "obs_trn": obs_trn,
            "obs_pred": obs_pred,
            "doy_trn": doy_trn,
            "doy_pred": doy_pred,
            "distance_matrix": dist_mat_out,
        }
    if out_file:
        np.savez_compressed(out_file, **data)
    return data

def get_data_lstm_da(data_file,
                     ar1_temp,
                     ar1_temp_pos,
                     doy_feat,
                     n_en,
):
    data = np.load(data_file) 

    if ar1_temp:
        seg_tave_water_mean = data['seg_tave_water_mean']
        seg_tave_water_std = data['seg_tave_water_std'] 
        obs_mean = data['obs_mean']
        obs_std = data['obs_std']
    
        # add in yesterday's water temperature as a driver (AR1)
        temp_minus1 = data['x_trn'][:,0:(data['x_trn'].shape[1]-1),ar1_temp_pos]
        x_trn = data['x_trn'][:,1:data['x_trn'].shape[1],:]
        x_trn[:,:,ar1_temp_pos] = temp_minus1
        y_trn = data['y_trn'][:,1:data['y_trn'].shape[1],:] 
        obs_trn = data['obs_trn'][:,1:data['obs_trn'].shape[1],:]
        obs_trn_ar1 = data['obs_trn'][:,0:(data['obs_trn'].shape[1]-1),:]
        doy_trn = data['doy_trn'][0:(data['doy_trn'].shape[0]-1)]
        doy_pred = data['doy_pred'][0:(data['doy_pred'].shape[0]-1)]
    else: 
        seg_tave_water_mean = float('NaN') 
        seg_tave_water_std = float('NaN')
        obs_mean = float('NaN')
        obs_std = float('NaN')
        
        x_trn = data['x_trn'] 
        y_trn = data['y_trn'] 
        obs_trn = data['obs_trn'] 
        obs_trn_ar1 = data['obs_trn'][:,0:(data['obs_trn'].shape[1]-1),:]
        doy_trn = data['doy_trn']
        doy_pred = data['doy_pred']
        
    if doy_feat:
        doy_trn = np.tile(doy_trn, n_en).reshape((n_en, doy_trn.shape[0],1))
        doy_pred = np.tile(doy_pred, n_en).reshape((n_en, doy_pred.shape[0],1))
        x_trn = np.append(x_trn, doy_trn, axis = 2)

    # get model prediction parameters for setting up EnKF matrices 
    if ar1_temp: 
        obs_array = data['obs_pred'][:,1:data['obs_pred'].shape[1],:] 
        temp_minus1 = data['x_pred'][:,0:(data['x_pred'].shape[1]-1),ar1_temp_pos] # this will be updated with DA 
        x_pred = data['x_pred'][:,1:data['x_pred'].shape[1],:]
        x_pred_da = data['x_pred'][:,1:data['x_pred'].shape[1],:]
        x_pred_f = data['x_pred'][:,1:data['x_pred'].shape[1],:]
        x_pred[:,:,ar1_temp_pos] = temp_minus1
        x_pred_da[:,:,ar1_temp_pos] = temp_minus1
        x_pred_f[:,:,ar1_temp_pos] = temp_minus1
        # add in observations if there are obs 
        scaled_obs = (data['obs_pred'][:,0:(data['obs_pred'].shape[1]-1),:] - obs_mean) / (obs_std + 1e-10)
        scaled_obs = np.repeat(scaled_obs, n_en, axis = 0)  
        x_pred[:,:,ar1_temp_pos] = np.where(np.isnan(scaled_obs[:,:,0]), x_pred[:,:,ar1_temp_pos], scaled_obs[:,:,0])
        x_pred_da[:,:,ar1_temp_pos] = np.where(np.isnan(scaled_obs[:,:,0]), x_pred_da[:,:,ar1_temp_pos], scaled_obs[:,:,0])
        x_pred_f[:,:,ar1_temp_pos] = np.where(np.isnan(scaled_obs[:,:,0]), x_pred_f[:,:,ar1_temp_pos], scaled_obs[:,:,0])
        
    else: 
        obs_array = data['obs_pred'] 
        x_pred = data['x_pred'] 
        x_pred_da = data['x_pred'] 
        x_pred_f = data['x_pred'] 
        
    if doy_feat:
        x_pred = np.append(x_pred, doy_pred, axis = 2)
    
    model_locations = data['model_locations'] # seg_id_nat of the stream segments 
    
    if ar1_temp: 
        dates = data['dates_pred'][1:data['dates_pred'].shape[0]]
        dates_trn = data['dates_trn'][1:data['dates_trn'].shape[0]]
    else: 
        dates = data['dates_pred'] 
        dates_trn = data['dates_trn'] 
    
    dist_mat = data['distance_matrix'] 
    
    return x_trn, y_trn, obs_trn, obs_trn_ar1, x_pred, x_pred_da, x_pred_f, obs_array, model_locations, dist_mat, dates, dates_trn, seg_tave_water_mean, seg_tave_water_std, obs_mean, obs_std
    
   
    
def sort_dist_matrix(mat, row_col_names):
    """
    sort the distance matrix by seg_id_nat
    :return:
    """
    df = pd.DataFrame(mat, columns=row_col_names, index=row_col_names)
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
    df = df.to_numpy()
    return df


def prep_adj_matrix(mat, rowcolnames, out_file=None):
    """
    process adj matrix.
    **The resulting matrix is sorted by seg_id_nat **
    :param infile:
    :param rowcolnames: row and column names of distance matrix 
    :param out_file:
    :return: [numpy array] processed adjacency matrix
    """
    
    adj = sort_dist_matrix(mat, rowcolnames)
    adj = np.where(np.isinf(adj), 0, adj)
    adj = -adj
    mean_adj = np.mean(adj[adj != 0])
    std_adj = np.std(adj[adj != 0])
    adj[adj != 0] = adj[adj != 0] - mean_adj
    adj[adj != 0] = adj[adj != 0] / std_adj
    adj[adj != 0] = 1 / (1 + np.exp(-adj[adj != 0]))

    I = np.eye(adj.shape[0])
    A_hat = adj.copy() + I
    D = np.sum(A_hat, axis=1)
    D_inv = D ** -1.0
    D_inv = np.diag(D_inv)
    A_hat = np.matmul(D_inv, A_hat)
    if out_file:
        np.savez_compressed(out_file, dist_matrix=A_hat)
    return A_hat


    