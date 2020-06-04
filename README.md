# Reinforcement Learning for determining the Re-order level

The goal of this repo is to teach an agent to determine the reorder level 
given a demand distribution, holding and inventory costs. The algorithm
used is a Deep Q-Learning algorithm with experience replay and a target
network. The deep-learning framework uses the nn-module of PyTorch.
The amount to reorder is currently determined manually by computing the
EOQ. Letting the agent determine the EOQ could be included later, but
with a more suitable algorithm for continuous action spaces
such as Policy Gradient.

