# Autonomous Car Parking Simulation with Behavioral Cloning 🚗🅿️

This repository contains a Python-based autonomous car parking system developed using **Supervised Learning** and **Behavioral Cloning**. Instead of using complex artificial neural network (ANN) libraries, the project focuses on training a custom agent based on human driving trajectories and sensor data mapping.

## 📁 Repository Structure
- `park_train.py`: The core Python script used to train the vehicle control agent and process training logs.
- `weights.npy`: The trained model weights representing the learned driving and parking behaviors.
- `history.txt`: The captured dataset containing human driving trajectories, control inputs, and state logs used during training.
- `project_final_presentation.pdf`: The official final presentation document detailing the project's architecture, methodology, and simulation results.

## 🚀 Key Features
- **Behavioral Cloning:** Successfully mapped human driving inputs directly to vehicle actuation commands (steering and throttle).
- **Custom Dataset Training:** Utilized state logs saved in `history.txt` to capture human-operated parking success lines.
- **Efficient Weight Management:** Trained weights are exported and loaded locally via NumPy (`weights.npy`) for immediate inference without retraining.

## 🛠️ Technologies & Tools Used
- **Language:** Python
- **Core Libraries:** NumPy (Data manipulation & matrix operations), Pandas, Matplotlib
- **Concepts:** Supervised Learning, Behavioral Cloning, Trajectory Tracking, Autonomous Vehicle Control

## 📊 Methodology & Presentation
For a detailed analysis of the simulation environment, data collection techniques, and mathematical background, please refer to the [project_final_presentation.pdf](project_final_presentation.pdf) file included in this repository.
