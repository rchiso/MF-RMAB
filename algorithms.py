import numpy as np
import random as rd
from copy import deepcopy
from utils import arm_indices


def MF_RMAB(c,env,epochs,n_episodes,episode_len,delta,optimal):
  n_arms, n_states, n_actions,_ = env.all_transitions.shape
  budget = env.budget
  total_reward = np.zeros(n_episodes*episode_len)
  timestep = 0
  for _ in range(epochs):
    min_matrix = np.ones((n_arms,n_states,n_actions))
    n_pulls = np.zeros((n_arms,n_states,n_actions)) #n_pulls[i,s,a] = No. of times action a was taken on arm i when it was at state s
    n_engaged = np.zeros((n_arms,n_states,n_actions)) #n_engaged[i,s,a] =  No. of times arm i went to state 1 after action a was taken at state s
    n_pulls_history = []
    n_engaged_history = []
    for ep in range(n_episodes):
      env.reset()
      P_hat = np.zeros((n_arms,n_states,n_actions)) #P_hat[i,s,a] = Probability of arm i going to state 1 after action a on state s
      radii = np.zeros((n_arms,n_states,n_actions))
      P_high = np.zeros((n_arms,n_states,n_actions))
      

      P_hat = n_engaged/np.maximum(min_matrix,n_pulls)

      radii = np.sqrt( 2 * n_states * np.log( 2 * n_states * n_actions * n_arms * ((ep+1)**4) / delta)  / np.maximum(min_matrix,n_pulls))

      P_high = np.minimum(min_matrix, P_hat + radii/2)

      indices, probs = arm_indices(n_arms,P_high,P_high,c)
      n_pulls_history.append(deepcopy(n_pulls))
      n_engaged_history.append(deepcopy(n_engaged))


      for h in range(episode_len):
        action = np.zeros(n_arms)
        if optimal == False:
          selection_idx = np.random.choice(a=np.arange(n_arms), size=budget, replace=False, p=probs)
        else:
          selection_idx = np.argpartition(indices, -1*budget)[-1*budget:]
      
        action[selection_idx] = 1
        action = action.astype(int)
        
        states = env.observe() #current states of all arms
        
        next_states, reward, _, _ = env.step(action)
        total_reward[timestep] = reward
        timestep += 1
        
        for i in range(n_arms):
          n_pulls[i][states[i]][action[i]] += 1
          if next_states[i] == 1:
            n_engaged[i][states[i]][action[i]] += 1
  
  
  n_pulls_history.append(n_pulls)
  return n_pulls_history, n_engaged_history, total_reward




def FAWT_Q(env,n_episodes,freq,L):
  #This code is for freq=2 only
  
  n_arms, n_states, n_actions,_ = env.all_transitions.shape
  L_ = L + n_arms
  n_pulls = np.zeros((n_arms,n_states,n_actions))
  z_pulls = np.zeros((n_arms,n_states,n_actions,L_))
  indices = np.zeros((n_arms,n_states,L_))
  Q = np.zeros((n_arms,n_states,n_actions,L_))
  rewards = [0,1]
  total_reward = np.zeros(n_episodes)
  last_pulled = -1*np.ones(n_arms)
  second_last_pulled = -1*np.ones(n_arms)
  last_pulled_history = []
  second_last_pulled_history = []
  l = np.zeros(n_arms)
  l = l.astype(int)
  l_history = []
  env.reset()
  for ep in range(n_episodes):
    old_l = deepcopy(l)
    k = env.budget
    states = env.observe()
    current_indices = np.zeros(n_arms)
    available_arms = np.arange(n_arms)
    chosen_arms = []
    violating_arms = np.where(l >= L)[0]
    for i in range(n_arms):
      current_indices[i] = indices[i,states[i],l[i]]
    for arm in violating_arms:
      if k > 0:
        chosen_arms.append(arm)
        k = k - 1
        available_arms = np.delete(available_arms, np.where(available_arms == arm))
        current_indices[arm] = -1*np.inf
    for i in range(n_arms):
      
      if i not in available_arms:
        continue
      if ep - second_last_pulled[i] >= L:
        if k > 0:
          chosen_arms.append(i)
          k = k - 1
          available_arms = np.delete(available_arms, np.where(available_arms == i))
          current_indices[i] = -1*np.inf
    selection_idx = []
    
    if k > 0:
      epsilon = n_arms/(n_arms+ep)
      if rd.random() < epsilon:
        selection_idx = np.random.choice(a=available_arms, size=k, replace=False)
      else:
        selection_idx = np.argpartition(current_indices, -1*k)[-1*k:]
    action = np.zeros(n_arms)
    for i in range(n_arms):
      if i in chosen_arms or i in selection_idx:
        action[i] = 1
        second_last_pulled[i] = last_pulled[i]
        last_pulled[i] = ep
        l[i] = 0
      else:
        l[i] += 1
    last_pulled_history.append(deepcopy(last_pulled))
    second_last_pulled_history.append(deepcopy(second_last_pulled))
    l_history.append(deepcopy(l))
    
    action = action.astype(int)
    
    next_states, r, _, _ = env.step(action)
    total_reward[ep] = r
    
    for i in range(n_arms):
      n_pulls[i][states[i]][action[i]] += 1
      z_pulls[i][states[i]][action[i]][l[i]] += 1

    for i in range(n_arms):
      alpha = 1/(1+z_pulls[i,states[i],action[i]][old_l[i]])
      Q[i,states[i],action[i],old_l[i]] = (1 - alpha)*Q[i,states[i],action[i],old_l[i]] + alpha*(rewards[states[i]]+ np.max(Q[i,next_states[i],:,l[i]]))
      indices[i,states[i],l[i]] = Q[i,states[i],1,l[i]] - Q[i,states[i],0,l[i]]
  return total_reward, n_pulls, z_pulls



