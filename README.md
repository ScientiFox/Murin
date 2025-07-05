![murin digest](https://github.com/user-attachments/assets/e17b00e6-2052-4159-a135-e0692d75083c)

<h2>Murin Algorithm</h2>

Murin is a machine learning algorithm originally developed in 2014 which is designed to learn sequential problem solving behaviors. It is inspired by operant conditioning, and primarily designed to enable the learning of behavior-based routines, in the sense of training a reactive system to respond to sensory values.

The algorithm, analysis thereof, and further date are all included in the following publication: https://ieeexplore.ieee.org/abstract/document/6950737

<h3>Mechanisms</h3>

To mimic operant conditioning for behavior training, the Murin algorithm is built on Q Learning, with two specific augmentations:

- Augmented states which are comprised of the observed state, and the previous action or state, to link together states in a sequential chain with a temporal, but relativistic mechanism
- A memory-based reinforcement mechanism, where augmented-state/action pairs in a historical queue are all reinforced together at time of reward application, with decaying rewards applied to older pairs, in order to reinforce sequential action sequences as a group

These mechanism together enable effective learning of sequential behavior series, and of identification of patterns over time and state spaces in tandem. The Murin agent is therefor able to successfully learn many problems which traditional reinforcement learning algorithms fail to solve, and often sustantially faster due to the overlapping rewards. Of special note, when the reward mechanisms encourage solutions, but do not provide negative reinforcement for unnecessary actions, the agents can learn what are, in essence, neuroses!


<h3>Training Methods</h3>
There are a wide range of training techniques which can be applied to Murin agents, especially due to their temporal characteristics. Some we use here are:

- **Rewards for exploration**: Several of the tests herein apply rewards primarily for the agent detecting a state which it has not yet observed, or for uniform spanning coverage od state observations, in order to encourage exploration and find goal states more effectively. This is demonstrated in the maze problems especially.
- **Rewards for state resolutions**: Rewarding the agent for changing deleterious states teaches it to effectively transition out of those states, and is an entirely internally driven reward mechanism. The robotic example implements this method.
- **Rewards for goal reaching**: The standard approach, when the goal is reached, preceeding state choices are rewarded. The Sussman Anomaly experiments implement this mechanism.
- **Rewards for metric improvement**: When an action, or sequence of actions, improve a metric measure for success, reward is applied, and possibly negative reinforement when the quality decreases. The Tower of Hanoi experiments implement this method.
- **Rewards for successful competition**: Reward for an agent out-performing another agent, such as another Murin, QL, or classical algorithm. The Tic Tac Toe and Maze experiments implement this adversarial learning (which I pitched to a professor fruitlessly in 2014. I'm still salty about that)

<h3>Contents</h3>

These files contain several experimental systems implementing the Murin algorithm and exploring its properties. The following problems are approached:
- **Grid Navigation**: The Murin agent learning to navigate on a grid, based on relative measures relating the current position to the destination. Notably, it only implements a relative cardinal direction (such as East, Northwest, and similar), and a distance reduction measure, rather than a full state space representing the grid space- the grid itself is abstractly limitless for the experiments.
- **Maze Navigation**: The Murin agent learning to navigate within a maze, based exclusively on observations of the local walls. This test demonstrates the learning of entirely local and relativistic observations, as well as learning in a complex and hierarchical sequential problem.
- **Tic Tac Toe**: The Murin agent learning to play Tic Tac Toe, based on adversarial learning. It's surprisingly good at it, too.
- **The Tower of Hanoi puzzle**: The Murin agent learning to solve the Tower of Hanoi puzzle, demonstrating learning on a highly regimented and hierarchical problem, with strict state space limits and a very large action space.
- **The Sussman Anomaly**: The Murin agent learning to solve the Sussman Anomaly, to demonstrate it is not constrained by it
- **Obstacle Avoidance**: A robotic implementation of Murin, learning Behavior-Based Robotics solutions to obstacle avoidance and unsticking, based on seeking to reward resolving those states when detected, as well a sreal tiem embedded learning.
