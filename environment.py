import numpy as np
import random as rd

### All credits to https://github.com/lily-x/online-rmab


def random_transition(all_population, n_states, n_actions, epsilon, random_seed):
    """ set initial transition probabilities
    returns array (N, S, A) that shows probability of transitioning to a **good** state """

    assert n_actions == 2
    np.random.seed(seed=random_seed)
    transitions = np.random.random((all_population, n_states, n_actions))
    transitions = np.maximum(epsilon*np.ones((all_population, n_states, n_actions)), transitions)

    transitions = np.minimum((1-epsilon)*np.ones((all_population, n_states, n_actions)), transitions)


    full_transitions = np.zeros((all_population, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = transitions
    full_transitions[:,:,:,0] = 1 - transitions

    return full_transitions


def assert_valid_transition(transitions):
    """ check that acting is always good, and starting in good state is always good """

    bad = False
    N, n_states, n_actions, _ = transitions.shape
    for i in range(N):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1,1] < transitions[i,s,0,1]:
                bad = True
                print(f'')
                #print(f'acting should always be good! {transitions[i,s,1,1]:.3f} < {transitions[i,s,0,1]:.3f}')

            # assert transitions[i,s,1,1] >= transitions[i,s,0,1] + 1e-6, f'acting should always be good! {transitions[i,s,1,1]:.3f} < {transitions[i,s,0,1]:.3f}'

    for i in range(N):
        for a in range(n_actions):
            # ensure start state is always good
            # assert transitions[i,1,a,1] >= transitions[i,0,a,1] + 1e-6, f'good start state should always be good! {transitions[i,1,a,1]:.3f} < {transitions[i,0,a,1]:.3f}'
            if transitions[i,1,a,1] < transitions[i,0,a,1]:
                bad = True
                print("b")
                #print(f'good start state should always be good! {transitions[i,1,a,1]:.3f} < {transitions[i,0,a,1]:.3f}')
    #assert bad == False


def random_valid_transition(all_population, n_states, n_actions):
    """ set initial transition probabilities
    returns array (N, S, A) that shows probability of transitioning to a **good** state

    enforce "valid" transitions: acting is always good, and starting in good state is always good """

    assert n_actions == 2

    transitions = np.random.random((all_population, n_states, n_actions))

    for i in range(all_population):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1] < transitions[i,s,0]:
                diff = 1 - transitions[i,s,0]
                transitions[i,s,1] = transitions[i,s,0] + (np.random.rand() * diff)

    for i in range(all_population):
        for a in range(n_actions):
            # ensure starting in good state is always good
            if transitions[i,1,a] < transitions[i,0,a]:
                diff = 1 - transitions[i,0,a]
                transitions[i,1,a] = transitions[i,0,a] + (np.random.rand() * diff)

    full_transitions = np.zeros((all_population, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = transitions
    full_transitions[:,:,:,0] = 1 - transitions

    # return transitions
    return full_transitions



def random_valid_transition_round_down(all_population, n_states, n_actions,epsilon,random_seed):
    """ set initial transition probabilities
    returns array (N, S, A) that shows probability of transitioning to a **good** state

    enforce "valid" transitions: acting is always good, and starting in good state is always good """

    assert n_actions == 2
    np.random.seed(seed=random_seed)
    transitions = np.random.random((all_population, n_states, n_actions))
    transitions = np.maximum(epsilon*np.ones((all_population, n_states, n_actions)), transitions)

    transitions = np.minimum((1-epsilon)*np.ones((all_population, n_states, n_actions)), transitions)


    for i in range(all_population):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1] < transitions[i,s,0]:
                transitions[i,s,0] = transitions[i,s,1] * np.random.rand()

    for i in range(all_population):
        for a in range(n_actions):
            # ensure starting in good state is always good
            if transitions[i,1,a] < transitions[i,0,a]:
                transitions[i,0,a] = transitions[i,1,a] * np.random.rand()


    full_transitions = np.zeros((all_population, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = transitions
    full_transitions[:,:,:,0] = 1 - transitions

    return full_transitions

def synthetic_transition_small_window(all_population, n_states, n_actions, low, high):
    """ set initial transition probabilities
    returns array (N, S, A) that shows probability of transitioning to a **good** state

    enforce "valid" transitions: acting is always good, and starting in good state is always good """

    assert n_actions == 2
    assert low < high
    assert 0 < low < 1
    assert 0 < high < 1

    transitions = np.random.random((all_population, n_states, n_actions))


    for i in range(all_population):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1] < transitions[i,s,0]:
                transitions[i,s,0] = transitions[i,s,1] * np.random.rand()

    for i in range(all_population):
        for a in range(n_actions):
            # ensure starting in good state is always good
            if transitions[i,1,a] < transitions[i,0,a]:
                transitions[i,0,a] = transitions[i,1,a] * np.random.rand()

    # scale down to a small window .4 to .6
    max_val = np.max(transitions)
    min_val = np.min(transitions)

    transitions = transitions - min_val
    transitions = transitions * (high - low) * (max_val - min_val) + low

    full_transitions = np.zeros((all_population, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = transitions
    full_transitions[:,:,:,0] = 1 - transitions

    return full_transitions


def CPAP(all_population, n_states, n_actions,epsilon,random_seed):

    assert n_actions == 2
    ratio = 0.3
    std = 0.1
    np.random.seed(seed=random_seed)
    adherence = np.array([[0.9615,1],[0.9743,1]])
    non_adherence = np.array([[0.2576,0.28336],[0.4278,0.47058]])
    transitions = np.zeros((all_population, n_states, n_actions))
    split = int(ratio*all_population)
    for i in range(split):
        np.random.seed(seed=(i+1)*random_seed)
        transitions[i] = non_adherence + np.random.normal(0,std,size=(n_states,n_actions))
    for i in range(split,all_population):
        np.random.seed(seed=(i+1)*random_seed)
        transitions[i] = adherence + np.random.normal(0,std,size=(n_states,n_actions))


    transitions = np.maximum(epsilon*np.ones((all_population, n_states, n_actions)), transitions)

    transitions = np.minimum((1-epsilon)*np.ones((all_population, n_states, n_actions)), transitions)


    full_transitions = np.zeros((all_population, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = transitions
    full_transitions[:,:,:,0] = 1 - transitions

    return full_transitions
