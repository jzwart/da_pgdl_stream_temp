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
):
    data = xr.open_dataset(var_file)  # open netcdf file with xarray 
    # make sure the indices are in the same order
    data = data.transpose("valid_time", "seg_id_nat", "ens")
    
    dates = data['valid_times']
    dates = dates.values
    dates = byte_to_str(dates)
    trn_date_idxs = np.array(np.where(np.isin(dates, [start_date_trn,end_date_trn]))).reshape(2).tolist()
    pred_date_idxs = np.array(np.where(np.isin(dates, [start_date_pred,end_date_pred]))).reshape(2).tolist()
    timedelta = pd.timedelta_range(start='0 days', end = '5112 days') # need to make this dynamic based on netcdf file
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
    if scale_data:
        return data_trn, data_pred, seg_tave_water_mean, seg_tave_water_std
    else: 
        return data_trn, data_pred 

def prep_one_var_lstm_da(
    var_file,
    vars,
    seg_id,
    start_date_trn,
    end_date_trn,
    start_date_pred,
    end_date_pred,
    scale_data=False,
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
    return data_trn, data_pred


def fmt_dataset(dataset):
    return dataset.to_array().values


def prep_data_lstm_da(
    obs_temp_file,
    driver_file,
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
):
    x_trn, x_pred, seg_tave_water_mean, seg_tave_water_std = get_nc_data(
        driver_file,
        x_vars,
        seg_id,
        start_date_trn,
        end_date_trn,
        start_date_pred,
        end_date_pred,
        scale_data=True,
    )
    y_trn, y_pred = get_nc_data(
        driver_file,
        y_vars,
        seg_id,
        start_date_trn,
        end_date_trn,
        start_date_pred,
        end_date_pred,
        scale_data=False,
    )
    obs_trn, obs_pred = prep_one_var_lstm_da(
        obs_temp_file,
        obs_vars,
        seg_id,
        start_date_trn,
        end_date_trn,
        start_date_pred,
        end_date_pred,
        scale_data=False,
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
    x_trn_out = np.empty((len(seg_id)*n_en, x_trn.shape[1], x_trn.shape[3])) # shape of [nseg*n_en, seq_len, nfeats]
    # the following should ensure that the first axis is sorted by seg_id_nat 
    for i in range(n_en):
        for j in range(len(seg_id)):
            cur_idx = i+j
            x_trn_out[cur_idx,:,:] = x_trn[j,:,i,:]
    #x_trn = np.repeat(x_trn, n_en, axis = 0)  
    # adding noise to predictors 
    #for i in range(x_trn.shape[0]):
        # should make this adjusted not by the scaled drivers 
     #   x_trn[i,:,:] = x_trn[i,:,:] + np.random.normal(scale = 0.2, size = x_trn.shape[1]).reshape((x_trn.shape[1],1))
    
    y_trn = fmt_dataset(y_trn) # current shape is [nfeats, seg_len, nseg, n_en]
    y_trn = np.moveaxis(y_trn, 0, -1)
    y_trn = np.moveaxis(y_trn, 1, 0)  # should now be in shape of [nseg, seq_len, n_en, nfeats]
    # we need to make the first axis repeated by n_en 
    y_trn_out = np.empty((len(seg_id)*n_en, y_trn.shape[1], y_trn.shape[3])) # shape of [nseg*n_en, seq_len, nfeats]
    # the following should ensure that the first axis is sorted by seg_id_nat 
    for i in range(n_en):
        for j in range(len(seg_id)):
            cur_idx = i+j
            y_trn_out[cur_idx,:,:] = y_trn[j,:,i,:]
    
    obs_trn = fmt_dataset(obs_trn)
    obs_trn = np.moveaxis(obs_trn, 0, -1)
    obs_trn = np.moveaxis(obs_trn, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    obs_trn = np.repeat(obs_trn, n_en, axis = 0)  
    
    x_pred = fmt_dataset(x_pred) # current shape is [nfeats, seg_len, nseg, n_en]
    x_pred = np.moveaxis(x_pred, 0, -1)
    x_pred = np.moveaxis(x_pred, 1, 0)  # should now be in shape of [nseg, seq_len, n_en, nfeats]
    # we need to make the first axis repeated by n_en 
    x_pred_out = np.empty((len(seg_id)*n_en, x_pred.shape[1], x_pred.shape[3])) # shape of [nseg*n_en, seq_len, nfeats]
    # the following should ensure that the first axis is sorted by seg_id_nat 
    for i in range(n_en):
        for j in range(len(seg_id)):
            cur_idx = i+j
            x_pred_out[cur_idx,:,:] = x_pred[j,:,i,:]
            
    y_pred = fmt_dataset(y_pred) # current shape is [nfeats, seg_len, nseg, n_en]
    y_pred = np.moveaxis(y_pred, 0, -1)
    y_pred = np.moveaxis(y_pred, 1, 0)  # should now be in shape of [nseg, seq_len, n_en, nfeats]
    # we need to make the first axis repeated by n_en 
    y_pred_out = np.empty((len(seg_id)*n_en, y_pred.shape[1], y_pred.shape[3])) # shape of [nseg*n_en, seq_len, nfeats]
    # the following should ensure that the first axis is sorted by seg_id_nat 
    for i in range(n_en):
        for j in range(len(seg_id)):
            cur_idx = i+j
            y_pred_out[cur_idx,:,:] = y_pred[j,:,i,:]
            
    obs_pred = fmt_dataset(obs_pred)
    obs_pred = np.moveaxis(obs_pred, 0, -1)
    obs_pred = np.moveaxis(obs_pred, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    
    data = {
        "x_trn": x_trn_out,
        "x_pred": x_pred_out,
        "dates_trn": dates_trn,
        "dates_pred": dates_pred,
        "model_locations": seg_id, 
        "y_trn": y_trn_out,
        "y_pred": y_pred_out,
        "obs_trn": obs_trn,
        "obs_pred": obs_pred,
        "seg_tave_water_mean": seg_tave_water_mean,
        "seg_tave_water_std": seg_tave_water_std,
        "doy_trn": doy_trn,
        "doy_pred": doy_pred,
    }
    if out_file:
        np.savez_compressed(out_file, **data)
    return data

