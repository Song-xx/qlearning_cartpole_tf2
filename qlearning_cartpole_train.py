"""
Usage:
    qlearning_cartpole_train.py [options]

Options:
    --hidden-size=<int>                 size of hidden state [default: 500]
    --max-iteration=<int>               number of iteration [default: 3000]
    --batch-size=<int>                  batch size [default: 40]
    --max-memory=<int>                  max length of memory queue [default: 2500]
    --lr=<int>                          learning rate [default: 0.007]
    --discount-rate=<float>             discount rate of q-value [default: 0.975]
    --epsilon-threshold=<int>           epsilon stay constant after it [default: 400]
    --model-save-every=<int>            save model frequency [default: 1]
    --model-save-path=<file>            model save path [default: ./]
"""

from docopt import docopt
import os

import tensorflow as tf
import gym
import numpy as np
from tensorflow.keras.layers import Dense
from collections import deque

game_name="CartPole-v0"
env=gym.make(game_name)
input_dim=4
n_outputs=2

tf.random.set_seed(10)

#### build a nn model
def nn(hidden_size=500):
    model=tf.keras.models.Sequential()
    model.add(Dense(hidden_size,activation='relu',input_dim=input_dim,kernel_initializer=tf.initializers.he_normal()))
    model.add(Dense(n_outputs,activation='relu',kernel_initializer=tf.initializers.he_normal()))
    return(model)

####  epsilon greedy policy
def epsilon_greedy_policy(state,epsilon=0):
    if np.random.random()<=epsilon:
        action=np.random.randint(2)
        return(action)
    else:
        Q_values=model.predict(state[np.newaxis])
        action=np.argmax(Q_values[0])
        return(action)

####
def play_one_step(env,state,epsilon):
    action=epsilon_greedy_policy(state,epsilon)
    next_state,reward,done,info=env.step(action)
    if not done:
        replay_buffer.append((state,action,reward,next_state,done))
    return(next_state,reward,done,info)

####
def sample_experiences(batch_size):
    indices=np.random.randint(len(replay_buffer),size=batch_size)
    experiences=[replay_buffer[index] for index in indices]
    states,actions,rewards,next_states,dones=[
        np.array([experience[i] for experience in experiences])
              for i in range(5)]
    return(states,actions,rewards,next_states,dones)

####
def training_step(batch_size,n_outputs=2,discount_factor=0.95):
    experiences=sample_experiences(batch_size)
    states,actions,rewards,next_states,dones=experiences
    #### calculate target q values
    next_q_values=model.predict(next_states)
    max_next_q_values=np.max(next_q_values,axis=1)
    target_q_values=(rewards+(1-dones)*discount_factor*max_next_q_values)
    #### calculate output q values
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_q_values = model(states)
        q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
        loss=tf.reduce_mean(loss_fn(target_q_values,q_values))
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))


if __name__ == '__main__':
    args=docopt(__doc__)
    ####
    save_path=args['--model-save-path']
    if not os.path.exists(save_path):
        assert os.path.exists(save_path)==True,'--model-save-path is NOT exists'
    ####
    hidden_size=int(args['--hidden-size'])
    n_iteration = int(args['--max-iteration'])
    max_momory = int(args['--max-memory'])
    lr=float(args['--lr'])
    discount_factor=float(args['--discount-rate'])
    batch_size=int(args['--batch-size'])
    epsilon_iteration_threshold=int(args['--epsilon-threshold'])
    save_freq=int(args['--model-save-every'])
    save_path=args['--model-save-path']
    ##
    print("HyperParameters:",lr,discount_factor,epsilon_iteration_threshold)
    ##
    replay_buffer=deque(maxlen=max_momory)
    model=nn(hidden_size)
    loss_fn=tf.losses.mean_squared_error
    optimizer=tf.optimizers.Adam(lr)
    ##
    for iteration in range(1,n_iteration+1):
        print("QLearning CartPole:iteration = %04d" % (iteration))
        epsilon = max(1-iteration/epsilon_iteration_threshold,0.01)
        obs=env.reset()
        while 1:
            obs,reward,done,info=play_one_step(env,obs,epsilon)
            if done:
                break
        if iteration>10:
            training_step(batch_size=batch_size,discount_factor=discount_factor)
        if iteration%save_freq==0:
            model.save(save_path+'%04d.h5'%(iteration))