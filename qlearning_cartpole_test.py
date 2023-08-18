"""
Usage:
    qlearning_cartpole_test.py [options]

Options:
    --model-save-path=<file>        h5 model save path [default: ./]
    --test-episodes=<int>           how many episodes to test the model [default: 100]
    --render=<bool>                 whether to render when playing game [default: False]
    --result-save-path=<file>       save the result into a csv [default: ./]
"""
from docopt import docopt
import os

import tensorflow as tf
import gym
from tensorflow.keras.layers import Dense
import numpy as np
import os
import pandas as pd
import time


def test_CartPole(env, n_episodes, render):
    sum_step = 0
    max_step_list = []
    for _ in range(n_episodes):
        obs = env.reset()
        for i in range(1, 210):
            q_value = model(obs[np.newaxis])
            action_i = tf.argmax(q_value, axis=1)
            obs, reward, done, info = env.step(int(action_i))
            if render:
                time.sleep(0.01)
                env.render()
            if done:
                print('CartPole episode %3d,max_step=%d' % (_, i))
                sum_step = sum_step+i
                max_step_list.append(i)
                env.close()
                break
    print('CartPole:mean max step =%.2f' % (sum_step/n_episodes))
    print('--------')
    print('\n')
    return ([sum_step/n_episodes]+max_step_list)


if __name__ == '__main__':
    args = docopt(__doc__)
    assert args['--model-save-path'] != "", '--model-save-path is NOT exists'
    assert args['--result-save-path'] != "", '--result-save-path is NOT exists'
    ##
    test_episode = int(args['--test-episodes'])
    render = eval(args['--render'])
    model_save_path = args['--model-save-path']
    result_save_path = args['--result-save-path']
    ##
    game_name = "CartPole-v0"
    env = gym.make(game_name)
    all_model = os.listdir(model_save_path)
    all_model.sort()
    all_result = []
    for model_ in all_model:
        if 'h5' in model_:
            print('model name : ', model_)
            model = tf.keras.models.load_model(model_save_path+model_)
            result = test_CartPole(env, n_episodes=test_episode, render=render)
            result_model_ = [model_[:-3]]+result
            all_result.append(result_model_)
    df = pd.DataFrame(all_result)
    df.columns = ['name']+['mean_max_step'] + \
        ["epi_%d" % x for x in range(1, test_episode+1)]
    df.sort_values(by='mean_max_step', ascending=False, inplace=True)
    df.to_csv(result_save_path+'qlearning_cartpole_test.csv', index=0)
