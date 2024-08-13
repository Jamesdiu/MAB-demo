def mab_sim(
    data, 
    experiment_id, 
    reward_type, reward_col='reward_revenue', variant_col = 'variant_id', batch_col = 'batch', strategy = 'thompson'):
    import pandas as pd
    import func_sampling_new_data as sampling
    import thompson_utils as tu

    mab_users = data.copy()
    mab_users['variant_id'] = mab_users[variant_col].astype(str)
    mab_users['batch'] = mab_users[batch_col].astype(str)
    weights_hist = pd.DataFrame()
    sampled_data_size = pd.DataFrame()

    # initialize weights
    variant_weights = {v: 1/mab_users.variant_id.nunique() for v in mab_users.variant_id.unique()}
    experiment_weight = {}
    experiment_weight[experiment_id] = variant_weights
    
    # initialize prior parameters
    prior_hist = {variant: {'alpha': 1, 'beta': 1} for variant in mab_users.variant_id.unique()}
    experiment_prior = {}
    experiment_prior[experiment_id] = prior_hist

    # make sure the batch is sorted in ascending order
    sorted_batch = sorted(mab_users.batch.unique())
    variant_id = mab_users.variant_id.unique()

    for batch in sorted_batch:

        # get all data from the batch
        batched_mab_users = mab_users[mab_users.batch == batch]
        # get variant weights from previous batch
        weights_hist[batch] = variant_weights
        # sample new data from the batch
        sampled_data = sampling.sampling_new_data(experiment_weight[experiment_id], batched_mab_users, batch)
        sampled_data_size[batch] = sampled_data['variant_id'].value_counts().to_dict()

        if strategy == 'thompson':
            variant_weights = tu.mab_thompson_variant_prob(
                experiment = experiment_id,
                mab_sample = sampled_data,  
                experiment_prior=experiment_prior,
                reward_type = reward_type, 
                reward_col=reward_col,
                variant_id = variant_id,
                batch=batch)
        
        experiment_weight[experiment_id] = variant_weights

    return weights_hist, sampled_data_size