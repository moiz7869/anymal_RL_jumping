## Introduction
This repo is a part of Term Project of class ME491 Learning-based Control at KAIST taught by professor Jemin Hwangbo.

The goal of this project is to design the reward function in raisim simulation so that (4-leg robot) Anymal learn to climb up the cylinder as high as possible.

This repo presents my solution, in which the robot could climb up 1.25491 meters (rank 7th / 38).

## Main idea
The main idea is to use curriculumn learning with many parts of reward function that encourages the robot to perform actions with different objectives such as going forward, jump, etc.

The details could be found in the report directory.

## Result
The robot could climb up 1.25491 meters (rank 7th / 38).

[Learning Iteration Video](https://youtu.be/oV66VFc71TQ)

## raisim_env_anymal

### How to use this repo
There is nothing to compile here. This only provides a header file for an environment definition. Read the instruction of raisimGym. 

### Dependencies
- raisimgym

### Run

1. Compile raisimgym with one of the environment files
2. run runner.py of the task
