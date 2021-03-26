# Introduction

The following outlines the project details for the second project submission, Continuous Control, for the Udacity Ud893 Deep Reinforcement Learning Nanodegree (DRLND).

# Getting Started

## The Environment

This project will use the [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher).

![Trained Agent](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

There are two versions of this environment - a single agent and a twenty agent version (each with its own copy of the environment). The second twenty agent version is useful for algorithms that use multiple (non-interactive parallel) copies of the same agent to distribute the task of gathering experience.

### Environment Solution

For the single agent implementation, the task is episodic and in order to solve the environment, the agent must attain an average score of +30 over 100 consecutive episodes.

This submission will solve the twenty agent version of the environment. In this case, the 20 agents must acheive an average score of +30 over 100 consecutive episodes in order to solve the environment. Specifically, after each episode, we add up the rewards that each agent received to get a final episodic score for each agent. These 20 potentially different episodic scores are then averaged, resulting in an overall average score over all agents for the episode. An overall average score of +30 over 100 consecutive episodes is considered as solving the environment.

## Install and Dependencies

The following instructions will help you set up the environment on your machine.

### Step 1 - Clone the Repository

All files required for running the project are in the main project directory. Note that a copy of the `python/` directory from the [DRLND](https://github.com/udacity/deep-reinforcement-learning#dependencies) which contains additional dependencies has also been included in the main project directory.

### Step 2 - Download the Unity Environment

Note that if your operating system is Windows (64-bit), the Unity environment is included for that OS in the main project directory and you can skip this section. If you're using a different operating system, download the file you require from one of the following links:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)

Then, place the file in the main project directory folder and unzip (or decompress) the file.

## Instructions

The [Report.md](Report.md) file is a project summary report which includes a decription of the implementation, learning algorithm(s), hyperparameters, neural net model architectures, reward/episode plots and ideas for future work. The summary report should be read first as it explains the order in which to run the project notebook. The `P2.ipynb` jupyter notebook provides the code for running and training the actual agent(s).
