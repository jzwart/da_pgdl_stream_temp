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
    data = data.loc[dict(seg_id_nat=[seg_id])]

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

    data = {
        "x_trn": fmt_dataset(x_trn),
        "x_pred": fmt_dataset(x_pred),
        "dates_trn": x_trn.date.values,
        "dates_pred": x_pred.date.values,
        "y_trn": fmt_dataset(y_trn),
        "y_pred": fmt_dataset(y_pred),
    }
    if out_file:
        np.savez_compressed(out_file, **data)
    return data


prep_data_lstm_da(
    "../../../drb-dl-model/data/in/obs_temp_full",
    "../../../drb-dl-model/data/in/uncal_sntemp_input_output",
    1573,
    "2011-06-01",
    "2012-06-01",
    "2012-06-02",
    "2012-07-02",
    out_file="../out/lstm_da_data_just_air_temp.npz",
)
