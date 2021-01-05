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


def prep_data_lstm_da(
    obs_temp_file,
    driver_file,
    seg_id,
    start_date_trn,
    end_date_trn,
    start_date_pred,
    end_date_pred,
    x_vars=["seg_tave_air"],
    y_vars=["temp_c"],
    out_file=None,
    n_en = 1, # number of ensembles for DA - creating n_en batches 
):
    x_trn, x_pred = prep_one_var_lstm_da(
        driver_file,
        x_vars,
        seg_id,
        start_date_trn,
        end_date_trn,
        start_date_pred,
        end_date_pred,
        scale_data=True,
    )
    y_trn, y_pred = prep_one_var_lstm_da(
        obs_temp_file,
        y_vars,
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
    for i in range(x_trn.shape[0]):
        # should make this adjusted not by the scaled drivers 
        x_trn[i,:,:] = x_trn[i,:,:] + np.random.normal(scale = 0.2, size = x_trn.shape[1]).reshape((x_trn.shape[1],1))
    
    y_trn = fmt_dataset(y_trn)
    y_trn = np.moveaxis(y_trn, 0, -1)
    y_trn = np.moveaxis(y_trn, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    y_trn = np.repeat(y_trn, n_en, axis = 0)  
    
    x_pred = fmt_dataset(x_pred)
    x_pred = np.moveaxis(x_pred, 0, -1)
    x_pred = np.moveaxis(x_pred, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    y_pred = fmt_dataset(y_pred)
    y_pred = np.moveaxis(y_pred, 0, -1)
    y_pred = np.moveaxis(y_pred, 1, 0)  # should now be in shape of [nseg, seq_len, nfeats]
    
    data = {
        "x_trn": x_trn,
        "x_pred": x_pred,
        "dates_trn": dates_trn,
        "dates_pred": dates_pred,
        "model_locations": seg_id, 
        "y_trn": y_trn,
        "y_pred": y_pred,
    }
    if out_file:
        np.savez_compressed(out_file, **data)
    return data

seg_ids = [1573] # needs to be a list of seg_ids (even if one segment)

prep_data_lstm_da(
    obs_temp_file = "5_pgdl_pretrain/in/obs_temp_full",
    driver_file = "5_pgdl_pretrain/in/uncal_sntemp_input_output",
    seg_id = seg_ids,
    start_date_trn = "2000-06-01",
    end_date_trn = "2012-06-01",
    start_date_pred = "2012-06-02",
    end_date_pred = "2013-06-02",
    out_file="5_pgdl_pretrain/in/lstm_da_data_just_air_temp.npz",
    n_en = 100
)
