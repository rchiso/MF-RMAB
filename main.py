import time
from experiment import run_benchmark, generate_exp_data
from generate_plots import run_plots, plot_reward_benchmark, plot_exposure_benchmark
import multiprocessing

def benchmarks():
    N = 5
    K = 1
    seeds = 30
    n_episodes = 1000
    horizon = 200


    processes = []
    for i in range(seeds):
        p = multiprocessing.Process(target=run_benchmark, args=[N,K,i,n_episodes,horizon])
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
    
    plot_exposure_benchmark(N,K,seeds) #Will cause errors for N > 10 (too many arms to display)
    plot_reward_benchmark(N,K,seeds)


def regret():
    #High values of N, seeds, episodes will take up a lot of storage (can go up to 10gb)
    N = 5
    K = 1
    seeds = 30
    n_episodes = 10000
    horizon = 200
    c = 3
    processes = []
    for i in range(1,seeds+1):
        p = multiprocessing.Process(target=generate_exp_data, args=[N,K,i,n_episodes,horizon,c])
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
    
    run_plots(N, K, seeds, n_episodes, horizon, c)



if __name__=='__main__':
    starttime = time.time()
    
    #benchmarks()
    regret()
    
    
    print(f'That took {time.time()-starttime} seconds')

