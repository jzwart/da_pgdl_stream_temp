import numpy as np
import xarray as xr
import pandas as pd 


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


def prep_data_rgcn_da(
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
        dist_mat_out[cur_start:cur_end, cur_start:cur_end] = dist_mat
        rowcolnames_out[cur_start:cur_end] = np.array(seg_idxs).reshape((len(seg_id)))
    
    dist_mat_out = prep_adj_matrix(mat = dist_mat_out, rowcolnames= rowcolnames_out)
    
    data = {
        "x_trn": x_trn,
        "x_pred": x_pred,
        "dates_trn": dates_trn,
        "dates_pred": dates_pred,
        "model_locations": seg_id, 
        "y_trn": y_trn,
        "y_pred": y_pred,
        "distance_matrix": dist_mat_out,
    }
    if out_file:
        np.savez_compressed(out_file, **data)
    return data



def sort_dist_matrix(mat, row_col_names):
    """
    sort the distance matrix by seg_id_nat
    :return:
    """
    df = pd.DataFrame(mat, columns=row_col_names, index=row_col_names)
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
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



