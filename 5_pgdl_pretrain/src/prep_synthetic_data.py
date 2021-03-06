import numpy as np
import xarray as xr


def check_if_finite(arr):
    assert np.isfinite(arr.to_array().values).all()


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


def prep_synthetic_data(
    obs_temp_file,
    synthetic_file,
    synthetic_obs_error,
    obs_freq, 
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
    x_trn, x_pred = prep_one_var_lstm_da(
        synthetic_file,
        x_vars,
        seg_id,
        start_date_trn,
        end_date_trn,
        start_date_pred,
        end_date_pred,
        scale_data=True,
    )
    y_trn, y_pred = prep_one_var_lstm_da(
        synthetic_file,
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
    
    dates_trn = x_trn.date.values
    dates_pred = x_pred.date.values
    
    # creating n_en training 
    x_trn = fmt_dataset(x_trn)
    x_trn = np.moveaxis(x_trn, 0, -1)
    x_trn = np.moveaxis(x_trn, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    x_trn = np.repeat(x_trn, n_en, axis = 0)  
    # adding noise to predictors 
    #for i in range(x_trn.shape[0]):
        # should make this adjusted not by the scaled drivers 
     #   x_trn[i,:,:] = x_trn[i,:,:] + np.random.normal(scale = 0.2, size = x_trn.shape[1]).reshape((x_trn.shape[1],1))
    
    y_trn = fmt_dataset(y_trn)
    y_trn = np.moveaxis(y_trn, 0, -1)
    y_trn = np.moveaxis(y_trn, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    # keeping only observations at obs frequency 
    not_obs_idx = np.where(np.remainder(np.arange(0, y_trn.shape[1]), obs_freq) !=0)
    y_trn[:,not_obs_idx,:] = np.NaN # turn all points on non-obs days into nan's 
    y_trn = np.repeat(y_trn, n_en, axis = 0)  
    y_true_trn = y_trn # storing 'true state' 
    for i in range(y_trn.shape[0]):  # adding noise to true state 
        y_trn[i,:,:] = y_trn[i,:,:] + np.random.normal(scale = synthetic_obs_error, size = y_trn.shape[1]).reshape((y_trn.shape[1],1))
    # adding same noise to true state for every ensemble  
    #y_trn[:] = y_trn[:] + np.random.normal(scale = synthetic_obs_error, size = y_trn.shape[0] * y_trn.shape[1]).reshape((y_trn.shape))
    
    x_pred = fmt_dataset(x_pred)
    x_pred = np.moveaxis(x_pred, 0, -1)
    x_pred = np.moveaxis(x_pred, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    y_pred = fmt_dataset(y_pred)
    y_pred = np.moveaxis(y_pred, 0, -1)
    y_pred = np.moveaxis(y_pred, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    y_true_pred = y_pred # storing 'true state' 
    for i in range(y_pred.shape[0]):  # adding noise to true state 
        y_pred[i,:,:] = y_pred[i,:,:] + np.random.normal(scale = synthetic_obs_error, size = y_pred.shape[1]).reshape((y_pred.shape[1],1))
    #y_pred[:] = y_pred[:] + np.random.normal(scale = synthetic_obs_error, size = y_pred.shape[0] * y_pred.shape[1]).reshape((y_pred.shape))
    
    # actual observations 
    obs_trn = fmt_dataset(obs_trn)
    obs_trn = np.moveaxis(obs_trn, 0, -1)
    obs_trn = np.moveaxis(obs_trn, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    obs_pred = fmt_dataset(obs_pred)
    obs_pred = np.moveaxis(obs_pred, 0, -1)
    obs_pred = np.moveaxis(obs_pred, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    
    data = {
        "x_trn": x_trn,
        "x_pred": x_pred,
        "dates_trn": dates_trn,
        "dates_pred": dates_pred,
        "model_locations": seg_id, 
        "y_trn": y_trn,
        "y_true_trn": y_true_trn,
        "y_pred": y_pred,
        "y_true_pred": y_true_pred,
        "obs_trn": obs_trn,
        "obs_pred": obs_pred,
    }
    if out_file:
        np.savez_compressed(out_file, **data)
    return data

