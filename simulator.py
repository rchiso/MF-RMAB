import gym
import numpy as np

### All credits to https://github.com/lily-x/online-rmab


class RMABSimulator(gym.Env):
    '''
    This simulator simulates the interaction with a set of arms with unknown transition probabilities
    but with additional side information. This setup is aligned with restless multi-armed bandit problems
    where we do not have repeated access to the same set of arms, but instead a set of new arms may
    arrive in the next iteration with side information transferable between different iiterations.

    The inputs of the simulator are listed below:

        all_population: the total number of arms in the entire population
        all_features: this is a numpy array with shape (all_population, feature_size)
        all_transitions: this is a numpy array with shape (all_population, 2, 2)
                        state (NE, E), action (NI, I), next state (E)
        cohort_size: the number of arms arrive per iteration as a cohort
        episode_len: the total number of time steps per episode iteration
        budget: the number of arms that can be pulled in a time step

    '''

    def __init__(self, all_population, all_transitions, cohort_size, episode_len, n_instances, n_episodes, budget,
            number_states=2):
        '''
        Initialization
        '''

        self.all_population  = all_population
        self.all_transitions = all_transitions
        self.cohort_size     = cohort_size
        self.budget          = budget
        self.number_states   = number_states
        self.episode_len     = episode_len
        self.n_episodes      = n_episodes   # total number of episodes per epoch
        self.n_instances     = n_instances  # n_epochs: number of separate transitions / instances
        self.transitions     = all_transitions

        #assert_valid_transition(all_transitions)

        # set up options for the multiple instances
        # track the first random initial state
        self.instance_count = 0
        self.episode_count  = 0
        self.timestep       = 0

        # track indices of cohort members
        self.cohort_selection  = np.zeros((n_instances, cohort_size)).astype(int)
        self.first_init_states = np.zeros((n_instances, n_episodes, cohort_size)).astype(int)
        for i in range(n_instances):
            self.cohort_selection[i, :] = np.random.choice(a=self.all_population, size=self.cohort_size, replace=False)
            #print('cohort', self.cohort_selection[i, :])
            for ep in range(n_episodes):
                self.first_init_states[i, ep, :] = self.sample_initial_states(self.cohort_size)

    def reset_all(self):
        self.instance_count = -1

        return self.reset_instance()

    def reset_instance(self):
        """ reset to a new environment instance """
        self.instance_count += 1

        # get new cohort members
        cohort_idx       = self.cohort_selection[self.instance_count, :]
        self.transitions = self.all_transitions[cohort_idx] # shape: cohort_size x n_states x 2 x n_states
        self.episode_count = 0

        # current state initialization
        self.timestep    = 0
        self.states      = self.first_init_states[self.instance_count, self.episode_count, :]  # np.copy??


        return self.observe()

    def reset(self):
        self.timestep      = 0

        self.states        = self.first_init_states[self.instance_count, self.episode_count, :]
        #print(f'instance {self.instance_count}, ep {self.episode_count}, state {self.states}')

        self.episode_count += 1
        return self.observe()

    def fresh_reset(self):
        '''
        This function resets the environment to start over the interaction with arms. The main purpose of
        this function is to sample a new set of arms (with number_arms arms) from the entire population.
        This corresponds to an episode of the restless multi-armed bandit setting but with different
        setup during each episode.

        This simulator also supports infinite time horizon by setting the episode_len to infinity.
        '''

        # Sampling
        sampled_arms     = np.random.choice(a=self.all_population, size=self.cohort_size, replace=False)
        self.transitions = self.all_transitions[sampled_arms] # shape: cohort_size x n_states x 2 x n_states

        # Current state initialization
        self.timestep    = 0
        self.states      = self.sample_initial_states(self.cohort_size)

        return self.observe()

    def sample_initial_states(self, cohort_size, prob=0.5):
        '''
        Sampling initial states at random.
        Input:
            cohort_size: the number of arms to be initialized
            prob: the probability of sampling 0 (not engaging state)
        '''

        states = np.random.choice(a=self.number_states, size=cohort_size, p=[prob, 1-prob])
        return states

    def is_terminal(self):
        if self.timestep >= self.episode_len:
            return True
        else:
            return False

    def observe(self):
        return self.states

    def step(self, action):
        #assert len(action) == self.cohort_size
        #assert np.sum(action) <= self.budget

        next_states = np.zeros(self.cohort_size)
        for i in range(self.cohort_size):
            prob = self.transitions[i, self.states[i], action[i], :]
            next_state = np.random.choice(a=self.number_states, p=prob)
            next_states[i] = next_state

        self.states = next_states.astype(int)
        self.timestep += 1

        reward = self.get_reward()
        done = self.is_terminal()

        # print(f'  action {action}, sum {action.sum()}, reward {reward}')

        return self.observe(), reward, done, {}

    def get_reward(self):
        return np.sum(self.states)
