import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import analyze_data
sns.set(style='white')

def run_plots(N,K,seeds, episodes, horizon, c):
    matrix = np.zeros((seeds,episodes))
    opt_matrix = np.zeros((seeds,episodes))
    t0_list = []
    G_list = []
    fail_list = []
    data_list = np.zeros(3)
    for i in range(1,seeds+1):
      n_pulls = np.load(f'Exp_data/Pulls/Pulls_N={N},K={K},seed={i}.npy')
      n_engaged = np.load(f'Exp_data/Engaged/Engaged_N={N},K={K},seed={i}.npy')
      all_transitions = np.load(f'Exp_data/TrueProb/TrueProb_N={N},K={K},seed={i}.npy')
      fr, t0, G_hist, fails = analyze_data(c, all_transitions,n_pulls,n_engaged,n_arms=N,episodes=episodes)
      matrix[i-1] = np.cumsum(fr)
      if t0 != []:
        t0_list.append(min(t0))
      else:
        t0_list.append(horizon)
      G_list.append(max(max(G_hist)))
      fail_list.append(np.mean(fails)/episodes)

      opt_n_pulls = np.load(f'Exp_data/Pulls/Opt_Pulls_N={N},K={K},seed={i}.npy')
      opt_n_engaged = np.load(f'Exp_data/Engaged/Opt_Engaged_N={N},K={K},seed={i}.npy')
      all_transitions = np.load(f'Exp_data/TrueProb/TrueProb_N={N},K={K},seed={i}.npy')
      opt_fr, t0, G_hist, fails = analyze_data(c, all_transitions,opt_n_pulls,opt_n_engaged,n_arms=N,episodes=episodes)
      opt_matrix[i-1] = np.cumsum(opt_fr)

    data_list[0] = sum(t0_list)/len(t0_list)

    data_list[1] = sum(G_list)/len(G_list)

    data_list[2] = sum(fail_list)/len(fail_list)

    print(f't_0={data_list[0]}, G={data_list[1]}')
    
    mean = np.mean(matrix,axis=0)
    std = np.std(matrix,axis=0)

    opt_mean = np.mean(opt_matrix,axis=0)
    opt_std = np.std(opt_matrix,axis=0)
    
    x = horizon*np.arange(episodes)
    temp = mean + std
    plt.plot(x, mean, 'b-', label='MF-RMAB')
    plt.fill_between(x, mean - std, mean + std, color='b', alpha=0.2)

    plt.plot(x, opt_mean, 'r--', label='Optimal')
    plt.fill_between(x, opt_mean - opt_std, opt_mean + opt_std, color='r', alpha=0.2)

    plt.legend()
    plt.ylim(0,max(temp))
    plt.title(f"N={N},K={K}")
    plt.xlabel("Timesteps")
    plt.ylabel("Regret")
    plt.savefig(f'Regret_N={N},K={K}.png')
    #plt.show()


def plot_exposure_benchmark(N,K,seeds):
    pulls = np.zeros((6,seeds,N))
    
    for i in range(seeds):
        pulls[:,i,:] = np.load(f'Benchmark/Pulls/Pulls_N={N},K={K},seed={i+1}.npy')

    sum = np.sum(np.mean(pulls[0,:],axis=0))

    xlabels = list(map(str, np.arange(1,N+1)))
    means = {
        'FaWT_Q, L=15': np.mean(pulls[0,:],axis=0)/sum, 'FaWT_Q, L=30': np.mean(pulls[1,:],axis=0)/sum,
        'FaWT_Q, L=50': np.mean(pulls[2,:],axis=0)/sum, 'MF-RMAB, c=1': np.mean(pulls[3,:],axis=0)/sum,
        'MF-RMAB, c=3': np.mean(pulls[4,:],axis=0)/sum, 'MF-RMAB, c=10': np.mean(pulls[5,:],axis=0)/sum,
    }
    x = np.arange(len(xlabels))  # the label locations
    width = 0.13  # the width of the bars
    multiplier = 0


    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(15,9)
    for attribute, measurement in means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        #ax.bar_label(rects, padding=3, fmt='%.2f')
        multiplier += 1

    ax.set_ylabel('Exposure',fontsize=20)
    ax.set_xlabel('Arm',fontsize=20)
    ax.set_title('Exposure of arms by different algorithms',fontsize=30)
    ax.set_xticks(x + 2.5*width, xlabels, fontsize=20)
    #ax.set_yticks(ax.get_yticks, fontsize=20)

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=20)
    leg.set_in_layout(True)
    #plt.tight_layout()
    plt.yticks(fontsize=20)
    plt.savefig(f'Exposure_N={N},K={K}.png')
    #plt.show()


def plot_reward_benchmark(N,K,seeds):
    rewards = np.zeros((6,seeds))

    for i in range(seeds):
        rewards[:,i] = np.load(f'Benchmark/Rewards/Rewards_N={N},K={K},seed={i+1}.npy')
    
    maxim = np.max(np.mean(rewards[:,:],axis=1),axis=0)

    plt.figure(figsize=(10,7))
    avg_r = {
        'FaWT_Q, \n L=15': np.mean(rewards[0,:],axis=0)/maxim, 'FaWT_Q, \n L=30': np.mean(rewards[1,:],axis=0)/maxim,
        'FaWT_Q, \n L=50': np.mean(rewards[2,:],axis=0)/maxim, 'MF-RMAB, \n c=1': np.mean(rewards[3,:],axis=0)/maxim,
        'MF-RMAB, \n c=3': np.mean(rewards[4,:],axis=0)/maxim, 'MF-RMAB, \n c=10': np.mean(rewards[5,:],axis=0)/maxim,
    }
    
    plt.bar(*zip(*avg_r.items()), width=0.3)
    plt.xlabel('Algorithms',fontsize=20)
    plt.ylabel('Total reward (normalized)',fontsize=20)
    plt.title('Total reward for different algorithms',fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'Rewards_N={N},K={K}')
    #plt.show()

