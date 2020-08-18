# qlearning_cartpole_tf2

# What is this
play CartPole to 200 step with Q-learning using TensorFlow2.

this code is refined from Aurelien Geron's <Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow>.

# What is in it
qlearning_cartpole_train.py:  
train models ,write these models in "model-save-path" folder,default save a model every one step,so one can easier find a good model;


qlearning_cartpole_test.py:   
test models ,read models  from "model-save-path" folder ,test these model with several episodes ,write a result csv in "result-save-path"folder;


result_csv_and_good_models:   
I run four times of the train and test code and record the result in csv.
several best model choosen from all result,all these model can play 200 steps;

curve_of_experiment_0.png:  
max mean step curve of the experiment_0 (default parameters).One can plot a curve with the result csv of the policygradient_cartpole_test.py .  


# How to use
## train
(1) make a folder to reserve the trained models  # default : ./


(2) python3 qlearning_cartpole_train.py    #default parameters is recommended

## test
python3 qlearning_cartpole_test.py

# other
(1) when testing , you can use "--render=True" to show the cartpole,but it will be slow.


(2) the model will saved every iteration(default),the example max-step curve shows below:


![image](https://github.com/Song-xx/qlearning_cartpole_tf2/blob/master/curve_of_experiment_0.png)

