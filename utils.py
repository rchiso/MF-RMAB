import numpy as np

def avg_cum_sum(x):
  x = np.cumsum(x)
  y = np.zeros(len(x))
  for i in range(len(x)):
    y[i] = x[i]/(i+1)
  return y

def g(x, c):
  return np.exp(c*x)

def arm_indices(n_arms,P_high,P_low,c):
  indices = np.zeros(n_arms)
  for i in range(n_arms):
    indices[i] = (P_high[i,0,1]/(1 - P_high[i,1,1] + P_high[i,0,1])) - (P_low[i,0,0]/(1 - P_low[i,1,0] + P_low[i,0,0]))
  probs = g(indices,c)
  probs /= sum(probs)
  return indices, probs



def true_information(all_transitions,n_arms, c):
  index = np.zeros(n_arms)
  prob = np.zeros(n_arms)
  for i in range(n_arms):
    P = all_transitions[i]
    index[i] = (P[0,1,1]/(1 - P[1,1,1] + P[0,1,1])) - (P[0,0,1]/(1 - P[1,0,1] + P[0,0,1]))
  prob = g(index, c)
  prob /= sum(prob)
  return index,prob

def calculate_regret(prob_history,true_prob):
  T = len(prob_history)
  fair_regret = np.zeros(T)
  for t in range(T):
    fair_regret[t] = np.linalg.norm(prob_history[t] - true_prob,ord=1)
  return fair_regret

def analyze_data(c, all_transitions, n_pulls_history, n_engaged_history, n_arms, episodes):
  trans_hist = []
  radii_history = []
  prob_history = []
  min_matrix = np.ones((n_arms,2,2))
  _, true_prob = true_information(all_transitions, n_arms, c)
  t_0 = []
  for t in range(episodes):
    n_pulls = n_pulls_history[t]
    n_engaged = n_engaged_history[t]
    radii = np.sqrt( 2 * 2 * np.log( 2 * 2 * 2 * n_arms * ((t+1)**4) / 0.01)  / np.maximum(min_matrix,n_pulls))
    radii_history.append(radii)
    trans = n_engaged/np.maximum(min_matrix,n_pulls)
    trans_hist.append(trans)


    P_high = np.minimum(min_matrix, trans + radii/2)
    P_low = np.maximum(min_matrix - 1, trans - radii/2)
    eta_1 = 0
    eta_2 = 0
    for i in range(n_arms):
      temp1 = P_high[i,1,1] - P_low[i,0,1]
      temp2 = P_high[i,1,0] - P_low[i,0,0]

      if temp1 > eta_1:
        eta_1 = temp1
      if temp2 > eta_2:
        eta_2 = temp2
    if eta_1 < 1 and eta_2 < 1:
      t_0.append(t+1)
    _, probs = arm_indices(n_arms, P_high, P_high, c)
    prob_history.append(probs)
  fr = calculate_regret(prob_history, true_prob)

  G_current = np.zeros(n_arms)
  G_hist = [[] for _ in range(n_arms)]
  streaks = [True for _ in range(n_arms)]
  for t in range(1,episodes):
    n_diff = np.where(n_pulls_history[t] - n_pulls_history[t-1] > 0, True, False)
    for i in range(n_arms):
      if False not in n_diff[i] and streaks[i]: #all state action pairs visited
        G_hist[i].append(G_current[i]+1)
        G_current[i] = 0
        streaks[i] = False
      elif False in n_diff[i]: #some state action pair not visited
        G_current[i] += 1
        streaks[i] = True
  fails = np.zeros(n_arms)
  for t in range(1,episodes):
    n_diff = np.where(n_pulls_history[t] - n_pulls_history[t-1] > 0, True, False)
    for i in range(n_arms):
      if False in n_diff[i]:
        fails[i] += 1
  for i in range(n_arms):
    if streaks[i] == True:
      G_hist[i].append(G_current[i]+1)
  return fr, t_0, G_hist, fails

