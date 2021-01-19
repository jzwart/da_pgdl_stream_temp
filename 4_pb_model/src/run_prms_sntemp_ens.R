
#
run_prms_sntemp_ens = function(nc_output_file_ind,
                               nc_param_file_ind,
                               start,
                               stop,
                               n_en,
                               vars,
                               model_run_loc,
                               orig_model_loc,
                               model_fabric_file = 'GIS/Segments_subset.shp'){

  for(n in seq_len(n_en)){
    print(sprintf('starting ensemble # %s of %s', n, n_en))
    # copy over original run files to temporary file location - overwrites any changes made to params / drivers
    copy_model_to_run_dir(model_run_loc = model_run_loc,
                          orig_model_loc = orig_model_loc)
    set_sntemp_output(output_names = vars$state,
                      model_run_loc = model_run_loc)

    model_fabric = sf::read_sf(file.path(model_run_loc, model_fabric_file))

    model_locations = tibble(seg_id_nat = as.character(model_fabric$seg_id_nat),
                             model_idx = as.character(model_fabric$model_idx)) %>%
      arrange(as.numeric(model_idx))

    #start = as.Date(start)
    #stop = as.Date(stop)

    # set parameters by drawing from nc file for current ensemble
    param_list = nc_params_get(nc_file_ind = nc_param_file_ind,
                               ens = n)
    update_sntemp_params(param_names = names(param_list),
                         updated_params = param_list,
                         model_run_loc = model_run_loc)

    # update drivers if varying drivers

    # run PRMS-SNTemp with current ensemble params and drivers
    run_sntemp(start = start,
               stop = stop,
               spinup = T,
               restart = T,
               save_ic = F,
               model_run_loc = model_run_loc,
               var_init_file = 'prms_ic.out',
               var_save_file = 'prms_ic.out')

    # gather prms-sntemp output
    model_out = gather_sntemp_output(model_run_loc = model_run_loc,
                                     model_output_file = 'output/stream_temp.out.nsegment',
                                     model_fabric_file = 'GIS/Segments_subset.shp',
                                     sntemp_vars = vars$state)

    print(sprintf('storing prms-sntemp ensemble # %s of %s in ncdf', n, n_en))
    # put model output into nc file
    nc_model_put(var_df = model_out,
                 var_names = vars$state,
                 ens = n, # current ensemble
                 nc_file_ind = nc_output_file_ind)
  }

}
