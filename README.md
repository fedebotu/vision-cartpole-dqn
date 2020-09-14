# Vision-Based CartPole with DQN
Implementation of the CartPole from OpenAI's Gym using only visual input 
for Reinforcement Learning control with Deep Q-Networks

<p align="center">
  <img src="https://github.com/Juju-botu/vision-cartpole-dqn/blob/save_model/stabilization">
</p>

***Author:*** Federico Berto

Thesis Project for University of Bologna;
Reinforcement Learning: a Preliminary Study on Vision-Based Control

A special thanks goes to gi`Adam Paszke <https://github.com/apaszke>`_, 
for a first implementation of the DQN algorithm with vision input in
the Cartpole-V0 environment from OpenAI Gym.
`Gym website <https://gym.openai.com/envs/CartPole-v0>`__.

The goal of this project is to design a control system for stabilizing a
Cart and Pole using Deep Reinforcement Learning, having only images as 
control inputs. We implement the vision-based control using the DQN algorithm
combined with Convolutional Neural Network for Q-values approximation.

The last two frames of the Cartpole are used as input, cropped and processed 
before using them in the Neural Network. In order to stabilize the training,
we use an experience replay buffer as shown in the paper "Playing Atari with
Deep Reinforcement Learning:
 <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>__.

Besides, a target network to further stabilize the training process is used.
make the training not converge, we set a threshold for stopping training
when we detect stable improvements: this way we learn optimal behavior
without saturation. 

<p align="center">
  <img src="https://github.com/Juju-botu/vision-cartpole-dqn/blob/save_model/resources/Cartpole-Architecture.png" alt="DQN Architecture" height="400">
</p>

The GUI is a handy tool for saving and loading trained models, and also for
training start/stop. Models and Graphs are saved in Vision_Carpole/save_model
and Vision_Cartpole/save_graph respectively.
<p align="center">
  <img src="https://github.com/fedebberto/Vision_Based_CartPole_DQN/blob/save_model/save_graph/Cartpole_Vision_Stop-142_LastEpNum-20.png?raw=true">
</p>
