

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
from scipy import stats
from tqdm import tqdm 
    

#dataset=pd.read_csv('clean_dataset/data_all_in.csv')
dataset=pd.read_csv('clean_dataset/data_only_0_or_1_inactive.csv')

# We keep active players
dataset=dataset[dataset['participant.label']!='INACTIVE'].copy()

# From now on, it will be useful to split the working dataset in two datasets,
# one for each treatment
dataset_every_round=dataset[dataset['session.code']=='every_round'].copy()
dataset_last_round=dataset[dataset['session.code']=='last_round'].copy()
dataset_every_round.drop(columns=['session.code'],inplace=True)
dataset_last_round.drop(columns=['session.code'],inplace=True)

# We compute which teams are in each treatment
teams_er=np.array(dataset_every_round.team.unique())
teams_lr=np.array(dataset_last_round.team.unique())

M_er=len(teams_er)
M_lr=len(teams_lr)

# We create an arraw containing the round number
round_number=np.linspace(1,16,16,dtype=int)

# We define our confidence level and the corresponding alpha level
c=0.95
alpha_level=(1-c)/2

# We compute the associated t-value
t_value_er=t.ppf(q=1-alpha_level,df=M_er-1)
t_value_lr=t.ppf(q=1-alpha_level,df=M_lr-1)

mu_er=np.zeros(16)
mu_lr=np.zeros(16)
errors_er=np.zeros(16)
errors_lr=np.zeros(16)
h1_1=np.zeros(M_er)
h1_2=np.zeros(M_er)
for r in round_number:
    
    # The maximum possible contribution by one team is 100*(number of active players)
    normalization_er=100*np.array(dataset_every_round[dataset_every_round['subsession.round_number']==r].groupby('team')['player.contribution'].size())
    normalization_lr=100*np.array(dataset_last_round[dataset_last_round['subsession.round_number']==r].groupby('team')['player.contribution'].size())
    x_er=np.array(dataset_every_round[dataset_every_round['subsession.round_number']==r].groupby('team')['player.contribution'].sum()/normalization_er)
    x_lr=np.array(dataset_last_round[dataset_last_round['subsession.round_number']==r].groupby('team')['player.contribution'].sum()/normalization_lr)
    
    # We compute the standard error with the sample variance
    # With ddof=1 we compute the unbiased estimator for the variance with small 
    # sample size
    # sample_variance=(M/(M-1))*((1/M)*np.sum(x*x)-sample_mean*sample_mean)
    std_error_er=np.sqrt(np.var(x_er,ddof=1)/M_er)
    std_error_lr=np.sqrt(np.var(x_lr,ddof=1)/M_lr)

    error_er=t_value_er*std_error_er
    error_lr=t_value_lr*std_error_lr
    
    mu_er[r-1]=np.mean(x_er)
    mu_lr[r-1]=np.mean(x_lr)
    errors_er[r-1]=error_er
    errors_lr[r-1]=error_lr


    # We evaluate if both treatments are significantly different in each round
    ks_stat,p_value=stats.ks_2samp(x_er, x_lr)
        
    # We store some values to test our hypotheses
    
    # Hypothesis 1
    if r==13: h1_1=x_er
    elif r>13: h1_2=h1_2+x_er
    
    # Hypothesis 2
    if r==13: h2_compare1=x_er
    elif r==14: h2_compare2=x_er
    elif r==15: h2_compare3=x_er
    elif r==16: h2_compare4=x_er
    
    # Hypothesis 3
    if r==13: h3_compare1=x_lr
    elif r==14: h3_compare2=x_lr
    elif r==15: h3_compare3=x_lr
    elif r==16: h3_compare4=x_lr
    
 
print('HYPOTHESIS 1')
print('')
mu_er1 = mu_er[12]
mu_er234 = mu_er[13]+mu_er[14]+mu_er[15]
error_mu_er1 = errors_er[12]
error_mu_er234 = errors_er[13]+errors_er[14]+errors_er[15]
print(f'Round 13: {np.round(mu_er1,2)}+-{np.round(error_mu_er1,2)}')
print(f'Rounds 14-15-16: {np.round(mu_er234,2)}+-{np.round(error_mu_er234,2)}')
print('Proportion of contributions in the first round relative to the rest of rounds')
value = mu_er1/(mu_er1+mu_er234)
print(f'Value: {np.round(value,2)}')
error = (1/(mu_er1+mu_er234)**2)*(mu_er1*error_mu_er234+mu_er234*error_mu_er1)
print(f'Error: {np.round(error,2)}')
print('')
print('')

def compare_samples(sample1, sample2, alpha=0.05):
    results = {}

    # Normality tests (Shapiro-Wilk)
    stat1, p1 = stats.shapiro(sample1)
    stat2, p2 = stats.shapiro(sample2)
    results['normality_sample1_p'] = np.round(p1,3)
    results['normality_sample2_p'] = np.round(p2,3)

    if p1 < alpha or p2 < alpha:
        # Use Kolmogorov-Smirnov test if not both are normally distributed
        ks_stat, ks_p = stats.ks_2samp(sample1, sample2)
        results['test'] = 'Kolmogorov-Smirnov'
        results['statistic'] = np.round(ks_stat,3)
        results['p_value'] = np.round(ks_p,3)
    else:
        # If both are normal, check for homogeneity of variances
        lev_stat, lev_p = stats.levene(sample1, sample2)
        results['levene_p'] = np.round(lev_p,3)
        if lev_p > alpha:
            # Student's t-test (equal variances)
            t_stat, t_p = stats.ttest_ind(sample1, sample2, equal_var=True)
            results['test'] = "Student's t-test"
        else:
            # Welch's t-test (unequal variances)
            t_stat, t_p = stats.ttest_ind(sample1, sample2, equal_var=False)
            results['test'] = "Welch's t-test"
        results['statistic'] = np.round(t_stat,3)
        results['p_value'] = np.round(t_p,3)

    return results

print('HYPOTHESIS 2')
print('')
results12 = compare_samples(h2_compare1, h2_compare2, alpha=0.05)
print(results12)
results23 = compare_samples(h2_compare2, h2_compare3, alpha=0.05)
print(results23)
results34 = compare_samples(h2_compare3, h2_compare4, alpha=0.05)
print(results34)

print('')
print('')
print('HYPOTHESIS 3')
print('')
results12 = compare_samples(h3_compare1, h3_compare2, alpha=0.05)
print(results12)
results23 = compare_samples(h3_compare2, h3_compare3, alpha=0.05)
print(results23)
results34 = compare_samples(h3_compare3, h3_compare4, alpha=0.05)
print(results34)


#%% 


def get_social_norms_dataset(dataset):
    e_expectations1=dataset[['participant.label',
                                  'participant.code',
                                  'player.empiricalExpectations1',
                                  'subsession.round_number',
                                  'session.code']].copy()
    e_expectations2=dataset[['participant.label',
                                  'participant.code',
                                  'player.empiricalExpectations2',
                                  'subsession.round_number',
                                  'session.code']].copy()
    e_expectations3=dataset[['participant.label',
                                  'participant.code',
                                  'player.empiricalExpectations3',
                                  'subsession.round_number',
                                  'session.code']].copy()
    e_expectations4=dataset[['participant.label',
                                  'participant.code',
                                  'player.empiricalExpectations4',
                                  'subsession.round_number',
                                  'session.code']].copy()
    
    contributions=dataset[['participant.label',
                                'participant.code',
                                'player.contribution',
                                'subsession.round_number',
                                'session.code']].copy()
    
    n_expectations1=dataset[['participant.label',
                                  'participant.code',
                                  'player.normativeExpectations1',
                                  'subsession.round_number',
                                  'session.code']].copy()
    n_expectations2=dataset[['participant.label',
                                  'participant.code',
                                  'player.normativeExpectations2',
                                  'subsession.round_number',
                                  'session.code']].copy()
    n_expectations3=dataset[['participant.label',
                                  'participant.code',
                                  'player.normativeExpectations3',
                                  'subsession.round_number',
                                  'session.code']].copy()
    n_expectations4=dataset[['participant.label',
                                  'participant.code',
                                  'player.normativeExpectations4',
                                  'subsession.round_number',
                                  'session.code']].copy()
    
    n_beliefs1=dataset[['participant.label',
                                  'participant.code',
                                  'player.personalNormativeBeliefs1',
                                  'subsession.round_number',
                                  'session.code']].copy()
    n_beliefs2=dataset[['participant.label',
                                  'participant.code',
                                  'player.personalNormativeBeliefs2',
                                  'subsession.round_number',
                                  'session.code']].copy()
    n_beliefs3=dataset[['participant.label',
                                  'participant.code',
                                  'player.personalNormativeBeliefs3',
                                  'subsession.round_number',
                                  'session.code']].copy()
    n_beliefs4=dataset[['participant.label',
                                  'participant.code',
                                  'player.personalNormativeBeliefs4',
                                  'subsession.round_number',
                                  'session.code']].copy()
    
    unquestioned_rounds=np.array([2,3,4,6,7,8,10,11,12,14,15,16])
    
    
    e_expectations1=e_expectations1[~e_expectations1['subsession.round_number'].isin(unquestioned_rounds)].copy()
    e_expectations2=e_expectations2[~e_expectations2['subsession.round_number'].isin(unquestioned_rounds)].copy()
    e_expectations3=e_expectations3[~e_expectations3['subsession.round_number'].isin(unquestioned_rounds)].copy()
    e_expectations4=e_expectations4[~e_expectations4['subsession.round_number'].isin(unquestioned_rounds)].copy()
    n_expectations1=n_expectations1[~n_expectations1['subsession.round_number'].isin(unquestioned_rounds)].copy()
    n_expectations2=n_expectations2[~n_expectations2['subsession.round_number'].isin(unquestioned_rounds)].copy()
    n_expectations3=n_expectations3[~n_expectations3['subsession.round_number'].isin(unquestioned_rounds)].copy()
    n_expectations4=n_expectations4[~n_expectations4['subsession.round_number'].isin(unquestioned_rounds)].copy()
    n_beliefs1=n_beliefs1[~n_beliefs1['subsession.round_number'].isin(unquestioned_rounds)].copy()
    n_beliefs2=n_beliefs2[~n_beliefs2['subsession.round_number'].isin(unquestioned_rounds)].copy()
    n_beliefs3=n_beliefs3[~n_beliefs3['subsession.round_number'].isin(unquestioned_rounds)].copy()
    n_beliefs4=n_beliefs4[~n_beliefs4['subsession.round_number'].isin(unquestioned_rounds)].copy()
    
    e_expectations2['subsession.round_number']=e_expectations2['subsession.round_number']+1
    e_expectations3['subsession.round_number']=e_expectations3['subsession.round_number']+2
    e_expectations4['subsession.round_number']=e_expectations4['subsession.round_number']+3
    n_expectations2['subsession.round_number']=n_expectations2['subsession.round_number']+1
    n_expectations3['subsession.round_number']=n_expectations3['subsession.round_number']+2
    n_expectations4['subsession.round_number']=n_expectations4['subsession.round_number']+3
    n_beliefs2['subsession.round_number']=n_beliefs2['subsession.round_number']+1
    n_beliefs3['subsession.round_number']=n_beliefs3['subsession.round_number']+2
    n_beliefs4['subsession.round_number']=n_beliefs4['subsession.round_number']+3
    
    e_expectations1.rename(columns={'player.empiricalExpectations1':'empirical_expectations'},inplace=True)
    e_expectations2.rename(columns={'player.empiricalExpectations2':'empirical_expectations'},inplace=True)
    e_expectations3.rename(columns={'player.empiricalExpectations3':'empirical_expectations'},inplace=True)
    e_expectations4.rename(columns={'player.empiricalExpectations4':'empirical_expectations'},inplace=True)
    n_expectations1.rename(columns={'player.normativeExpectations1':'normative_expectations'},inplace=True)
    n_expectations2.rename(columns={'player.normativeExpectations2':'normative_expectations'},inplace=True)
    n_expectations3.rename(columns={'player.normativeExpectations3':'normative_expectations'},inplace=True)
    n_expectations4.rename(columns={'player.normativeExpectations4':'normative_expectations'},inplace=True)
    n_beliefs1.rename(columns={'player.personalNormativeBeliefs1':'normative_beliefs'},inplace=True)
    n_beliefs2.rename(columns={'player.personalNormativeBeliefs2':'normative_beliefs'},inplace=True)
    n_beliefs3.rename(columns={'player.personalNormativeBeliefs3':'normative_beliefs'},inplace=True)
    n_beliefs4.rename(columns={'player.personalNormativeBeliefs4':'normative_beliefs'},inplace=True)
    
    e_expectations=pd.concat([pd.concat([pd.concat([e_expectations1,e_expectations2]),e_expectations3]),e_expectations4])
    n_expectations=pd.concat([pd.concat([pd.concat([n_expectations1,n_expectations2]),n_expectations3]),n_expectations4])
    n_beliefs=pd.concat([pd.concat([pd.concat([n_beliefs1,n_beliefs2]),n_beliefs3]),n_beliefs4])
    
    df1=pd.merge(e_expectations,contributions,on=['participant.label','participant.code', 'subsession.round_number', 'session.code']).dropna()
    df2=pd.merge(n_expectations,n_beliefs,on=['participant.label','participant.code', 'subsession.round_number', 'session.code']).dropna()
    
    social_norms=pd.merge(df1,df2,on=['participant.label','participant.code', 'subsession.round_number', 'session.code']).dropna()

    return social_norms


def get_partners_of_p(p, teams):
    G=teams[teams['participant.code']==p].team.unique()[0]
    members=teams[teams.team==G]['participant.code'].unique()
    partners=members[members!=p]
    return partners

def get_EE_pt(social_norms,p,t):
    EE_p=social_norms[social_norms['participant.code']==p]
    EE_pt=EE_p[EE_p['subsession.round_number']==t]['empirical_expectations']
    return EE_pt

def get_NE_pt(social_norms,p,t):
    NE_p=social_norms[social_norms['participant.code']==p]
    NE_pt=NE_p[NE_p['subsession.round_number']==t]['normative_expectations']
    return NE_pt

def get_NB_pt(social_norms,p,t):
    NB_p=social_norms[social_norms['participant.code']==p]
    NB_pt=NB_p[NB_p['subsession.round_number']==t]['normative_beliefs']
    return NB_pt

def get_C_pt(social_norms,p,t):
    C_p=social_norms[social_norms['participant.code']==p]
    C_pt=C_p[C_p['subsession.round_number']==t]['player.contribution']
    return C_pt

def compute_consistency(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    for p in players: 

        EE_pt=get_EE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        NE_pt=get_NE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        
        sum_q=0
        max_sum_q=0
        partners=get_partners_of_p(p, teams)
        for q in partners:
            EE_qt=get_EE_pt(social_norms,q,t).reset_index(drop=True).loc[0]
            NE_qt=get_NE_pt(social_norms,q,t).reset_index(drop=True).loc[0]
            sum_q=sum_q+abs(EE_pt-EE_qt)+abs(NE_pt-NE_qt)
            max_sum_q=max_sum_q+100+100
            
        sum_p=sum_p+sum_q
        max_sum_p=max_sum_p+max_sum_q
        
    consistency=1.-sum_p/max_sum_p
    return consistency

def compute_empirical_consistency(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    for p in players: 

        EE_pt=get_EE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        
        sum_q=0
        max_sum_q=0
        partners=get_partners_of_p(p, teams)
        for q in partners:
            EE_qt=get_EE_pt(social_norms,q,t).reset_index(drop=True).loc[0]
            sum_q=sum_q+abs(EE_pt-EE_qt)
            max_sum_q=max_sum_q+100+100
            
        sum_p=sum_p+sum_q
        max_sum_p=max_sum_p+max_sum_q
        
    empirical_consistency=1.-sum_p/max_sum_p
    return empirical_consistency

def compute_normative_consistency(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    for p in players: 

        NE_pt=get_NE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        
        sum_q=0
        max_sum_q=0
        partners=get_partners_of_p(p, teams)
        for q in partners:
            NE_qt=get_NE_pt(social_norms,q,t).reset_index(drop=True).loc[0]
            sum_q=sum_q+abs(NE_pt-NE_qt)
            max_sum_q=max_sum_q+100+100
            
        sum_p=sum_p+sum_q
        max_sum_p=max_sum_p+max_sum_q
        
    normative_consistency=1.-sum_p/max_sum_p
    return normative_consistency


def compute_accuracy(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    M=0
    for p in players: 
    
        EE_pt=get_EE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        NE_pt=get_NE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        
        sum_q=0
        max_sum_q=0
        partners=get_partners_of_p(p, teams)
        for q in partners:
            C_qt=get_C_pt(social_norms,q,t).reset_index(drop=True).loc[0]
            NB_qt=get_NB_pt(social_norms,q,t).reset_index(drop=True).loc[0]
            sum_q=sum_q+abs(EE_pt-C_qt)+abs(NE_pt-NB_qt)
            max_sum_q=max_sum_q+100+100
            M=M+1
            
            
        sum_p=sum_p+sum_q
        max_sum_p=max_sum_p+max_sum_q
        
    accuracy=1.-sum_p/max_sum_p
    return accuracy, M

def compute_empirical_accuracy(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    for p in players: 
    
        EE_pt=get_EE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        
        sum_q=0
        max_sum_q=0
        partners=get_partners_of_p(p, teams)
        for q in partners:
            C_qt=get_C_pt(social_norms,q,t).reset_index(drop=True).loc[0]

            sum_q=sum_q+abs(EE_pt-C_qt)
            max_sum_q=max_sum_q+100
            
        sum_p=sum_p+sum_q
        max_sum_p=max_sum_p+max_sum_q
        
    empirical_accuracy=1.-sum_p/max_sum_p
    return empirical_accuracy

def compute_normative_accuracy(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    for p in players: 
    
        NE_pt=get_NE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        
        sum_q=0
        max_sum_q=0
        partners=get_partners_of_p(p, teams)
        for q in partners:
            NB_qt=get_NB_pt(social_norms,q,t).reset_index(drop=True).loc[0]
            sum_q=sum_q+abs(NE_pt-NB_qt)
            max_sum_q=max_sum_q+100
            
        sum_p=sum_p+sum_q
        max_sum_p=max_sum_p+max_sum_q
        
    normative_accuracy=1.-sum_p/max_sum_p
    return normative_accuracy


def compute_specificity(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    for p in players: 

        EE_pt=get_EE_pt(social_norms,p,t).reset_index(drop=True).loc[0]
        NE_pt=get_NE_pt(social_norms,p,t).reset_index(drop=True).loc[0]

        meanEE=EE_pt
        meanNE=NE_pt
        partners=get_partners_of_p(p, teams)
        for q in partners:
            meanEE=meanEE+get_EE_pt(social_norms,q,t).reset_index(drop=True).loc[0]
            meanNE=meanNE+get_NE_pt(social_norms,q,t).reset_index(drop=True).loc[0]
        
        meanEE=meanEE/(1+len(partners))
        meanNE=meanNE/(1+len(partners))
            
        sum_p=sum_p+abs(EE_pt-meanEE)+abs(NE_pt-meanNE)
        max_sum_p=max_sum_p+100+100
        
    specificity=1.-sum_p/max_sum_p
    return specificity

def compute_empirical_specificity(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    for p in players: 

        EE_pt=get_EE_pt(social_norms,p,t).reset_index(drop=True).loc[0]

        meanEE=EE_pt
        partners=get_partners_of_p(p, teams)
        for q in partners:
            meanEE=meanEE+get_EE_pt(social_norms,q,t).reset_index(drop=True).loc[0]
        
        meanEE=meanEE/(1+len(partners))
            
        sum_p=sum_p+abs(EE_pt-meanEE)
        max_sum_p=max_sum_p+100+100
        
    empirical_specificity=1.-sum_p/max_sum_p
    return empirical_specificity

def compute_normative_specificity(t,social_norms, players, teams):
    sum_p=0
    max_sum_p=0
    for p in players: 

        NE_pt=get_NE_pt(social_norms,p,t).reset_index(drop=True).loc[0]

        meanNE=NE_pt
        partners=get_partners_of_p(p, teams)
        for q in partners:
            meanNE=meanNE+get_NE_pt(social_norms,q,t).reset_index(drop=True).loc[0]
        
        meanNE=meanNE/(1+len(partners))
            
        sum_p=sum_p+abs(NE_pt-meanNE)
        max_sum_p=max_sum_p+100+100
        
    normative_specificity=1.-sum_p/max_sum_p
    return normative_specificity

#%% 




def compute_variables(treatment, dataset):
    dataset=dataset[dataset['participant.label']!='INACTIVE'].copy()
    social_norms=get_social_norms_dataset(dataset)
    social_norms=social_norms[social_norms['session.code']==treatment]
    rounds=social_norms.groupby('participant.code').size()
    social_norms=social_norms[~social_norms['participant.code'].isin(rounds[rounds<16].index)]

    teams=dataset[['participant.code','team']]
    teams=teams[teams['participant.code'].isin(social_norms['participant.code'])]
    
    players=np.array(social_norms['participant.code'].unique())

    time=np.arange(1,17,1)
    
    consistency=np.zeros(len(time))
    empirical_consistency=np.zeros(len(time))
    normative_consistency=np.zeros(len(time))
    
    accuracy=np.zeros(len(time))
    empirical_accuracy=np.zeros(len(time))
    normative_accuracy=np.zeros(len(time))
    
    specificity=np.zeros(len(time))
    empirical_specificity=np.zeros(len(time))
    normative_specificity=np.zeros(len(time))
    
    norm_strength=np.zeros(len(time))
    
    for t in tqdm(time): 
        
        consistency[t-1]=compute_consistency(t,social_norms, players, teams)
        empirical_consistency[t-1]=compute_empirical_consistency(t,social_norms, players, teams)
        normative_consistency[t-1]=compute_normative_consistency(t,social_norms, players, teams)
    
        accuracy[t-1], M=compute_accuracy(t,social_norms, players, teams)
        empirical_accuracy[t-1]=compute_empirical_accuracy(t,social_norms, players, teams)
        normative_accuracy[t-1]=compute_normative_accuracy(t,social_norms, players, teams)
    
        specificity[t-1]=compute_specificity(t,social_norms, players, teams)
        empirical_specificity[t-1]=compute_empirical_specificity(t,social_norms, players, teams)
        normative_specificity[t-1]=compute_normative_specificity(t,social_norms, players, teams)
        
        norm_strength[t-1]=consistency[t-1]*accuracy[t-1]*specificity[t-1]
        
    
    from scipy.stats import t
    c=0.95
    alpha_level=(1-c)/2
    t_value=t.ppf(q=1-alpha_level,df=M-1)

    error_c=t_value*np.sqrt((consistency*(1.-consistency))/M)
    error_ec=t_value*np.sqrt((empirical_consistency*(1.-empirical_consistency))/M)
    error_nc=t_value*np.sqrt((normative_consistency*(1.-normative_consistency))/M)
    
    error_a=t_value*np.sqrt((accuracy*(1.-accuracy))/M)
    error_ea=t_value*np.sqrt((empirical_accuracy*(1.-empirical_accuracy))/M)
    error_na=t_value*np.sqrt((normative_accuracy*(1.-normative_accuracy))/M)
    
    error_s=t_value*np.sqrt((specificity*(1.-specificity))/M)
    error_es=t_value*np.sqrt((empirical_specificity*(1.-empirical_specificity))/M)
    error_ns=t_value*np.sqrt((normative_specificity*(1.-normative_specificity))/M)
    
    error_n=t_value*np.sqrt((norm_strength*(1.-norm_strength))/M)
        
    return consistency, empirical_consistency, normative_consistency, accuracy, empirical_accuracy, normative_accuracy, specificity, empirical_specificity, normative_specificity, norm_strength,  error_c,  error_ec, error_nc, error_a,  error_ea, error_na, error_s, error_es, error_ns, error_n

consistency_er, empirical_consistency_er, normative_consistency_er, accuracy_er, empirical_accuracy_er, normative_accuracy_er, specificity_er, empirical_specificity_er, normative_specificity_er, norm_strength_er,  error_c_er,  error_ec_er, error_nc_er, error_a_er,  error_ea_er, error_na_er, error_s_er, error_es_er, error_ns_er, error_n_er = compute_variables('every_round', dataset)
consistency_lr, empirical_consistency_lr, normative_consistency_lr, accuracy_lr, empirical_accuracy_lr, normative_accuracy_lr, specificity_lr, empirical_specificity_lr, normative_specificity_lr, norm_strength_lr,  error_c_lr,  error_ec_lr, error_nc_lr, error_a_lr,  error_ea_lr, error_na_lr, error_s_lr, error_es_lr, error_ns_lr, error_n_lr = compute_variables('last_round', dataset)
time=np.arange(1,17,1)

#%% 

plt.figure('Norm Strength')
plt.plot(time,norm_strength_er, 's',color='cornflowerblue', alpha=0.7, label='Every Round')
plt.plot(time,norm_strength_lr, 's',color='firebrick', alpha=0.7, label='Last Round')

plt.plot(time,norm_strength_er, '-',color='cornflowerblue', alpha=0.7)
plt.plot(time,norm_strength_lr, '-',color='firebrick', alpha=0.7)

plt.fill_between(time,norm_strength_er+error_n_er, norm_strength_er-error_n_er,color='cornflowerblue', alpha=0.2)
plt.fill_between(time,norm_strength_lr+error_n_lr, norm_strength_lr-error_n_lr,color='firebrick', alpha=0.2)

plt.axvline(4.5,color='black')
plt.axvline(8.5,color='black')
plt.axvline(12.5,color='black')

plt.xlabel('Round number', fontsize=16)
plt.ylabel('Norm strength', fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title('Norm Strength Analysis', fontweight='bold')

plt.legend(fontsize=10, loc='lower right')
plt.grid()
plt.tight_layout()




#%%

plt.figure('Hypothesis 4a')
plt.plot(time,consistency_er, 'o',color='cornflowerblue', alpha=0.7, label='Every Round')
plt.plot(time,consistency_lr, 'o',color='firebrick', alpha=0.7, label='Last Round')

plt.plot(time,consistency_er, '-',color='cornflowerblue', alpha=0.7)
plt.plot(time,consistency_lr, '-',color='firebrick', alpha=0.7)

plt.fill_between(time,consistency_er+error_c_er, consistency_er-error_c_er,color='cornflowerblue', alpha=0.2)
plt.fill_between(time,consistency_lr+error_c_lr, consistency_lr-error_c_lr,color='firebrick', alpha=0.2)

plt.axvline(4.5,color='black')
plt.axvline(8.5,color='black')
plt.axvline(12.5,color='black')

plt.xlabel('Round number', fontsize=16)
plt.ylabel('Consistency', fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title('Hypothesis 4a: CONSISTENCY OF EXPECTATIONS', fontweight='bold')

plt.legend(fontsize=10, loc='lower left')
plt.grid()
plt.tight_layout()
plt.show()

#%% 

plt.figure('Hypothesis 4b.1')

plt.plot(time,empirical_accuracy_er, 'o',color='cornflowerblue', alpha=0.7, label='Every Round')
plt.plot(time,empirical_accuracy_lr, 'o',color='firebrick', alpha=0.7, label='Last Round')

plt.plot(time,empirical_accuracy_er, '-',color='cornflowerblue', alpha=0.7)
plt.plot(time,empirical_accuracy_lr, '-',color='firebrick', alpha=0.7)

plt.fill_between(time,empirical_accuracy_er+error_ea_er, empirical_accuracy_er-error_ea_er,color='cornflowerblue', alpha=0.2)
plt.fill_between(time,empirical_accuracy_lr+error_ea_lr, empirical_accuracy_lr-error_ea_lr,color='firebrick', alpha=0.2)

plt.axvline(4.5,color='black')
plt.axvline(8.5,color='black')
plt.axvline(12.5,color='black')

plt.xlabel('Round number', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title('Hypothesis 4b.1: ACCURACY OF EMPIRICAL EXPECTATIONS', fontweight='bold')

plt.legend(fontsize=10, loc='lower left')
plt.grid()
plt.tight_layout()

#%% 

plt.figure('Hypothesis 4b.2')

plt.plot(time,normative_accuracy_er, 'o',color='cornflowerblue', alpha=0.7, label='Every Round')
plt.plot(time,normative_accuracy_lr, 'o',color='firebrick', alpha=0.7, label='Last Round')

plt.plot(time,normative_accuracy_er, '-',color='cornflowerblue', alpha=0.7)
plt.plot(time,normative_accuracy_lr, '-',color='firebrick', alpha=0.7)

plt.fill_between(time,normative_accuracy_er+error_na_er, normative_accuracy_er-error_na_er,color='cornflowerblue', alpha=0.2)
plt.fill_between(time,normative_accuracy_lr+error_na_lr, normative_accuracy_lr-error_na_lr,color='firebrick', alpha=0.2)

plt.axvline(4.5,color='black')
plt.axvline(8.5,color='black')
plt.axvline(12.5,color='black')

plt.xlabel('Round number', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title('Hypothesis 4b.2: ACCURACY OF NORMATIVE EXPECTATIONS', fontweight='bold')

plt.legend(fontsize=10, loc='lower left')
plt.grid()
plt.tight_layout()

