import numpy as np
from simulator import RMABSimulator
from environment import random_transition ,random_valid_transition_round_down, CPAP
from algorithms import MF_RMAB, FAWT_Q

def generate_exp_data(n_arms,budget,seed_no,n_episodes,horizon, c):
    all_population  = n_arms
    number_states   = 2
    number_actions  = 2
    c               = c
    epsilon         = 0.01
    delta           = 0.01
    cohort_size     = all_population
    episode_len     = horizon
    budget          = budget
    n_instances     = 1
    n_episodes      = n_episodes

    #Synthetic-alternate
    all_transitions = random_valid_transition_round_down(all_population,number_states,number_actions,epsilon=epsilon, random_seed=n_arms*budget*seed_no)
    
    #Synthetic
    #all_transitions = random_transition(all_population,number_states,number_actions,epsilon=epsilon, random_seed=n_arms*budget*seed_no)
    
    #CPAP
    #all_transitions = CPAP(all_population, number_states, number_actions, epsilon=epsilon, random_seed=n_arms*budget*seed_no)

    #MF-RMAB
    simulator = RMABSimulator(all_population, all_transitions, cohort_size, episode_len,n_instances,n_episodes, budget, number_states)
    pull_history, engaged_history, _ = MF_RMAB(c, env=simulator, epochs = n_instances, n_episodes=n_episodes, episode_len=episode_len, delta=delta, optimal=False)
    np.save(f'Exp_data/Pulls/Pulls_N={n_arms},K={budget},seed={seed_no}',pull_history)
    np.save(f'Exp_data/Engaged/Engaged_N={n_arms},K={budget},seed={seed_no}',engaged_history)
    np.save(f'Exp_data/TrueProb/TrueProb_N={n_arms},K={budget},seed={seed_no}',all_transitions)
    
    #Optimal
    simulator = RMABSimulator(all_population, all_transitions, cohort_size, episode_len,n_instances,n_episodes, budget, number_states)
    opt_pull_history, opt_engaged_history, _ = MF_RMAB(c, env=simulator, epochs = n_instances, n_episodes=n_episodes, episode_len=episode_len, delta=delta, optimal=True)
    np.save(f'Exp_data/Pulls/Opt_Pulls_N={n_arms},K={budget},seed={seed_no}',opt_pull_history)
    np.save(f'Exp_data/Engaged/Opt_Engaged_N={n_arms},K={budget},seed={seed_no}',opt_engaged_history)
    print(f'Trial {seed_no} for N={n_arms},K={budget} done!')


def run_benchmark(N, K, i, n_episodes, horizon):
    all_population  = N
    number_states   = 2
    number_actions  = 2
    cohort_size     = all_population
    episode_len     = horizon
    budget          = K
    n_instances     = 1
    n_episodes      = n_episodes
    delta           = 0.01
    epsilon         = 0.01


    pulls = np.zeros((6, all_population))
    avg_r = np.zeros(6)
    temp_pulls = np.zeros((6,all_population))


    #Synthetic-alternate
    all_transitions = random_valid_transition_round_down(all_population,number_states,number_actions,epsilon=epsilon, random_seed=N*budget*i)
    
    #Synthetic
    #all_transitions = random_transition(all_population,number_states,number_actions,epsilon=epsilon, random_seed=n_arms*budget*seed_no)
    
    #CPAP
    #all_transitions = CPAP(all_population, number_states, number_actions, epsilon=epsilon, random_seed=n_arms*budget*seed_no)

    
    
    #FAWT-Q
    simulator = RMABSimulator(all_population, all_transitions, cohort_size, episode_len,n_instances,n_episodes, budget, number_states)
    reward_history1, n_pulls1, z_pulls1 = FAWT_Q(env=simulator, n_episodes=n_episodes*episode_len,freq=2,L=15)
    simulator = RMABSimulator(all_population, all_transitions, cohort_size, episode_len,n_instances,n_episodes, budget, number_states)
    reward_history2, n_pulls2, z_pulls2 = FAWT_Q(env=simulator, n_episodes=n_episodes*episode_len,freq=2,L=30)
    simulator = RMABSimulator(all_population, all_transitions, cohort_size, episode_len,n_instances,n_episodes, budget, number_states)
    reward_history3, n_pulls3, z_pulls3 = FAWT_Q(env=simulator, n_episodes=n_episodes*episode_len,freq=2,L=50)

    for j in range(all_population):
        temp_pulls[0,j] = np.sum(n_pulls1[j,:,1])
        temp_pulls[1,j] = np.sum(n_pulls2[j,:,1])
        temp_pulls[2,j] = np.sum(n_pulls3[j,:,1])
    pulls[0] = np.sort(temp_pulls[0])
    avg_r[0] = sum(reward_history1)
    pulls[1] = np.sort(temp_pulls[1])
    avg_r[1] = sum(reward_history2)
    pulls[2] = np.sort(temp_pulls[2])
    avg_r[2] = sum(reward_history3)
    




    #MF-RMAB
    optimal = False
    simulator = RMABSimulator(all_population, all_transitions, cohort_size, episode_len,n_instances,n_episodes, budget, number_states)
    n_pulls_history4, n_engaged_history4, reward_history4 = MF_RMAB(c = 1, env=simulator, epochs = n_instances, n_episodes=n_episodes, episode_len=episode_len, delta=delta, optimal=optimal)
    simulator = RMABSimulator(all_population, all_transitions, cohort_size, episode_len,n_instances,n_episodes, budget, number_states)
    n_pulls_history5, n_engaged_history5, reward_history5 = MF_RMAB(c = 3, env=simulator, epochs = n_instances, n_episodes=n_episodes, episode_len=episode_len, delta=delta, optimal=optimal)
    simulator = RMABSimulator(all_population, all_transitions, cohort_size, episode_len,n_instances,n_episodes, budget, number_states)
    n_pulls_history6, n_engaged_history6, reward_history6 = MF_RMAB(c = 10, env=simulator, epochs = n_instances, n_episodes=n_episodes, episode_len=episode_len, delta=delta, optimal=optimal)



    for j in range(all_population):
        temp_pulls[3,j] = np.sum(n_pulls_history4[-1][j,:,1])
        temp_pulls[4,j] = np.sum(n_pulls_history5[-1][j,:,1])
        temp_pulls[5,j] = np.sum(n_pulls_history6[-1][j,:,1])
    pulls[3] = np.sort(temp_pulls[3])
    avg_r[3] = sum(reward_history4)
    pulls[4] = np.sort(temp_pulls[4])
    avg_r[4] = sum(reward_history5)
    pulls[5] = np.sort(temp_pulls[5])
    avg_r[5] = sum(reward_history6)
    
    print(f'Trial {i} for N={N},K={K} done!')

    np.save(f'Benchmark/Pulls/Pulls_N={N},K={K},seed={i+1}',pulls)
    np.save(f'Benchmark/Rewards/Rewards_N={N},K={K},seed={i+1}',avg_r)