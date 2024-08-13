import pandas as pd

def mab_sim(
    data, reward_type, reward_col, variant_col, sample_col, batch_col, 
    experiment_prior = {}, strategy = 'thompson'
    ):

    # Define asnd standardise key columns
    mab_users = data.copy()
    mab_users['variant_id'] = mab_users[variant_col].astype(str)
    mab_users['batch'] = mab_users[batch_col].astype(str)
    mab_users['sample_size_col'] = mab_users[sample_col].fillna(0).astype(int)
    sorted_batch = sorted(mab_users.batch.unique()) # make sure the batch is sorted in ascending order
    variant_id = mab_users.variant_id.unique()

    weights_hist = pd.DataFrame()
    sampled_data_size = pd.DataFrame()
    params_list = {}

    # # initialize prior parameters
    if not experiment_prior:
        print('initialize prior parameters')
        prior_hist = {variant: {'alpha': 1, 'beta': 1} for variant in mab_users.variant_id.unique()}
        experiment_prior = {}
        experiment_prior = prior_hist

    # initialize even weights for all variants
    initial_variant_weights = {v: 1/mab_users.variant_id.nunique() for v in mab_users.variant_id.unique()}

    # starting weights for each experiment
    experiment_weight = initial_variant_weights

    for batch in sorted_batch:        
        batched_mab_users = mab_users[mab_users.batch == batch] # get all data from the batch
        weights_hist[batch] = experiment_weight # get variant weights from previous batch

        # sample new data from the batch
        sampled_data = sampling_new_data(experiment_weight, batched_mab_users, sample_col, batch)
        sampled_data_size[batch] = {variant_id: sample_size for variant_id, sample_size in zip(sampled_data['variant_id'], sampled_data['sample_size_col'])}
        if strategy == 'thompson':
            new_variant_weights, alpha, beta = mab_thompson_variant_prob(
                mab_sample = sampled_data,  
                experiment_prior=experiment_prior,
                reward_type = reward_type, 
                reward_col=reward_col,
                variant_id = variant_id,
                batch=batch)
        
        # update the variant weights for the next batch
        experiment_weight = new_variant_weights
        params_list[batch] = {'alpha': alpha, 'beta': beta}

        # New format initialization
        prior_dict = {}
        # Iterate through the original dictionary to populate the new format
        for param, campaigns in params_list[batch].items():
            for campaign, value in campaigns.items():
                if campaign not in prior_dict:
                    prior_dict[campaign] = {}
                prior_dict[campaign][param] = value  # Use the actual value instead of a fixed value

        experiment_prior = prior_dict
    return weights_hist, sampled_data_size, params_list

def mab_thompson_variant_prob(mab_sample, experiment_prior, reward_type, reward_col, variant_id, batch = 'None'):
    """
        Parameters:
        reward_col should be rate or per unit value
    """
    from gamma import GammaExponential, GammaPoisson
    from beta import BetaBinomial

    continuous_rewards = ['Revenue', 'cpi', 'Cost per Purchase', 'PurchaseRate']
    discrete_rewards = ['PaymentCount']
    bernoulli_rewards = ['ConversionRate']

    # 1.1. get new record, filtered with experiment, from the database, mab_sample, as input to run python script
    batched_df = mab_sample[mab_sample['batch'] == batch].copy()
    batched_df['reward_value'] = batched_df[reward_col].astype(float)*batched_df['sample_size_col']
    # new data inforamtion
    variant_stats = batched_df.groupby(variant_id).agg(
        reward_total = pd.NamedAgg(column='reward_value', aggfunc='sum'),
        sample_size = pd.NamedAgg(column='sample_size_col', aggfunc='sum'))

    # 1.3. update prior for each variant
    samples_dict = pd.DataFrame()
    last_alpha = {}
    last_beta = {}
    for variant in variant_id:
        # 1.2. get the parameters of the prior for each variant
        alpha = experiment_prior[variant]['alpha']
        beta = experiment_prior[variant]['beta']
        
        if reward_type in continuous_rewards: # contiuos reward
            prior_model = GammaExponential(alpha, beta)
            updated_model = prior_model.update(
            variant_stats.loc[variant, 'sample_size'],
            variant_stats.loc[variant, 'reward_total']
            )
        elif reward_type in discrete_rewards: # discrete reward
            prior_model = GammaPoisson(alpha, beta)
            updated_model = prior_model.update(
            variant_stats.loc[variant, 'reward_total'],
            variant_stats.loc[variant, 'sample_size']
            )
        elif reward_type in bernoulli_rewards: # Buroneulli reward
            prior_model = BetaBinomial(alpha, beta)
            print('not yet tested')

        else:
            print('reward_type not found. Supported reward types are:')
            print(continuous_rewards + discrete_rewards + bernoulli_rewards)
            break
        # update prior with new parameters
        # print('Group: ', variant,'alpha: ', updated_model.alpha, 'beta: ', updated_model.beta)

    # 1.4. draw 1000 samples from the posterior for each variant and update as next prior
        samples_dict[variant] = updated_model.sample(1000)
        last_alpha[variant] = updated_model.alpha
        last_beta[variant] = updated_model.beta

    # 1.5. calculate the probability of each variant being the best as their weight
    if ('cost' in reward_type.lower()) or ('cpi' in reward_type.lower()):
        best_variant = samples_dict.idxmin(axis=1)
    else:
        best_variant = samples_dict.idxmax(axis=1)
    variant_weights = best_variant.value_counts(normalize=True).sort_index()
    variant_weights = variant_weights.reindex(variant_id, fill_value=0)
    
    return variant_weights.to_dict(), last_alpha, last_beta
    
def sampling_new_data(variant_prob, mab_sample, sample_col='None', batch = 'None', n_samples = 100000):
    """
        Sample new data based on the probability of each variant being selected.
        Parameters:
        ----------
        variant_prob : dict
            The probability of each variant being selected
            - index : variant_id
            - value : probability
        mab_sample : DataFrame
            The DataFrame of samples from the MAB
        batch : str
            The batch of samples to sample from. Chronological order of the samples.
        n_samples : int, optional
            The target number of total samples to be drawn (default is 3000)

        Returns: 
        ----------
        DataFrame
            samples : The sampled DataFrame

    """

    # Initialize an empty DataFrame to store the samples
    samples = pd.DataFrame()
    batch_sample = mab_sample[mab_sample['batch'] == batch]
 
    for variant_id in batch_sample.variant_id.unique():   
        variant_sample = batch_sample[batch_sample.variant_id == variant_id]

        # check the amount of impression available for the variant. Max sample size = total impression
        n_variant_samples = int(n_samples * variant_prob[variant_id])
        if n_variant_samples < variant_sample[sample_col].sum():
            variant_sample['sample_size_col'] = n_variant_samples
        samples = pd.concat([samples, variant_sample], axis=0)
            
    if samples['sample_size_col'].sum() != n_samples:
        print(f'Waring: The sample size, {samples.sample_size_col.sum()} is not equal to the target sample size, {n_samples}.')

    return samples

def scenario_gen(df, reward_type, reward_col, experiment_prior, batch_col, 
            iteration = 100, variant_col = 'Campaign Name',
            sample_col = '# of Impressions', strategy='thompson'):
    weight_group = {}
    data_size_group = {}
    params_list = {}
    i = 0
    while i < iteration:
        weight_group[i], data_size_group[i], params_list[i] = mab_sim(
            data = df,
            reward_type=reward_type,
            reward_col=reward_col,
            variant_col = variant_col,
            sample_col = sample_col,
            experiment_prior = experiment_prior,
            batch_col = batch_col,
            strategy=strategy
            )
        i += 1
    plot_result(weight_group, df, reward_type, reward_col)
    return weight_group, data_size_group, params_list

def plot_result(weight_group, df, reward_type, reward_col):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_group = pd.DataFrame()
    for key in weight_group.keys():
        trans_weight_group = weight_group[key].sort_index().transpose().reset_index()
        all_group = pd.concat([all_group, trans_weight_group], axis=0)

    all_group['index'] = all_group['index'].astype(str)
    # Convert index to datetime to ensure proper date handling and sorting
    all_group.index = pd.to_datetime(all_group.index)
    comp_table = df.groupby(['Date','Campaign Name'])[reward_col].sum().sort_index().unstack()
    cost_per_purchase_ratio = comp_table['Test Campaign']/comp_table['Control Campaign']
    cost_per_purchase_ratio = pd.DataFrame(cost_per_purchase_ratio, columns=[f'{reward_type} Ratio']).reset_index()
    cost_per_purchase_ratio['index'] = cost_per_purchase_ratio.Date.astype(str)

    all_group = pd.concat([
        all_group.groupby('index').mean(), 
        cost_per_purchase_ratio.set_index('index')
        ], axis = 1)
        
    # Create a subplot with 2 y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the mean values line plot for Control and Test Campaigns on the primary y-axis
    for column in ['Control Campaign', 'Test Campaign']:
        fig.add_trace(go.Scatter(x=all_group.index, y=all_group[column], name=f'Mean {column}'), secondary_y=False)

    colors = all_group[f'{reward_type} Ratio'].apply(lambda x: 'green' if x > 1 else 'grey')
    # Add the cost per purchase ratio scatter plot on the secondary-axis
    fig.add_trace(go.Scatter(x=all_group.index, y=all_group[f'{reward_type} Ratio'], mode='markers', name='Test:Control Ratio', marker =dict(color=colors)), secondary_y=True)

    # Set x-axis properties
    fig.update_xaxes(title_text="Date", tickangle=-45, tickvals=all_group.index[::10], tickformat="%Y-%m-%d")

    # Set y-axes titles
    fig.update_yaxes(title_text="Mean Values", secondary_y=False)
    fig.update_yaxes(title_text=f"{reward_type} Ratio", secondary_y=True)
    fig.update_layout(title_text="Campaign Analysis", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Show plot
    fig.show()