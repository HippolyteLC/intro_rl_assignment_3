# literature

**curriculum learning** 
Curriculum learning falls inthe domain of transfer learning. Trabnsfer learning is the learning of an agent on a source task to then transition the learned policy, reward function, or model  in the target task. Curriculum learning is the automatic sequencding of source tasks in such a way that an agent can have better performance on a target task; here increased learning speed and performance is the goal {narvekar2016source} {narvekar2017autonomous}. Given the domain of the problem, a subset of learning tasks (source tasks) can be defined for the agent to train on. Randomized curriculum learning, i.e. a curriculum where source tasks are assigned randomly without any given heuristic can delay convergence by adding redundant tasks in the learning sequence {narvekar2017autonomous}. 

**Prioritized Experience Replay** 
Experience replayes allows an agent to reuse rare and useful prior experiences instead of learning from a given state transition once {lin1992self}. An experience is often denoted as a quadruple (st,a,r,s'). The agent selects an action a in a current state s for which it obtains reward r and transitions to the next state s'. Prioritized experience replay emphasies this further, by selecting which experiences to learn from based on how valuable that experience is in the context of the task. This extended method of experience replay can speed up the learning process of the agent (by a factor of 2) {schaul2015prioritized}. 





**strategy: strati replay buffer**

**strategy: least visited curriculum**

**strategy: worst performing curriculum**

# references

@article{lin1992self,
  title={Self-improving reactive agents based on reinforcement learning, planning and teaching},
  author={Lin, Long-Ji},
  journal={Machine learning},
  volume={8},
  number={3},
  pages={293--321},
  year={1992},
  publisher={Springer}
}

@article{schaul2015prioritized,
  title={Prioritized experience replay},
  author={Schaul, Tom and Quan, John and Antonoglou, Ioannis and Silver, David},
  journal={arXiv preprint arXiv:1511.05952},
  year={2015}
}

@inproceedings{narvekar2016source,
  title={Source task creation for curriculum learning},
  author={Narvekar, Sanmit and Sinapov, Jivko and Leonetti, Matteo and Stone, Peter},
  booktitle={Proceedings of the 2016 international conference on autonomous agents \& multiagent systems},
  pages={566--574},
  year={2016}
}

@inproceedings{narvekar2017autonomous,
  title={Autonomous Task Sequencing for Customized Curriculum Design in Reinforcement Learning.},
  author={Narvekar, Sanmit and Sinapov, Jivko and Stone, Peter},
  booktitle={IJCAI},
  pages={2536--2542},
  year={2017}
}
