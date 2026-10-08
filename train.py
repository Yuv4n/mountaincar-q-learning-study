#Objective is to get cart to flag
#Initially Just move Randomly
import gym
import numpy as np

env = gym.make("MountainCar-v0")
#Constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000


#Every 2000 loops shows programme is still running
SHOW_EVERY = 2000


#You might not know the Environment's Variables
#You will almost always know these actions

#New State value can be anything (Position and Velocity)
#New states given as continuous values (Too much storage to create table
#Convert New State value from Continuous Data to Discrete Data

#Dont hard code this as some environments have over 2 observational values
#20 (x) is number of chunks/groups, the Q learning table's number of dimensions is dependant on the variables in the environment
#Getting the value for x can take hundreds of trial and error attempts
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high) - (env.observation_space.low) / DISCRETE_OS_SIZE

#Epsilon is the amount of randomness from 0-1, 1 being the most random
# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
#// always gives an integer
END_EPSILON_DECAYING =  EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#Low and High change between each programmes action and its correlating reward
#Creating the randomized Q table
q_table = np.random.uniform(low = -2, high = 0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table



for episode in range(EPISODES):

    # If we can divide episodes by SHOW_EVERY without a remainder (mod)
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False



    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:


        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            #To track how many episodes it took to complete the track
            #F string print is just beautifull code
            print(f"We made it on episode {episode}")
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
