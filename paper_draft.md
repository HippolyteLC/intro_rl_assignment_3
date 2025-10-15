# Introduction
  
Deep Q learning is a powerful reinforcement learning algorithm used in many studies to solve problems where the state space is extremely vast or continuous; too large for tabular learning {mnih2015human}. This study will utilize the DQL algorithm to play the cartpole game from OpenAI's Gym environment. The state space of cartpole is continuous, therefore the application of DQL for this problem is suitable. Alongside a vanilla implementation of DQL, our baseline, we will also implement three alternate strategies for training our DQL algorithm. We evaluate our policy based on its generalized performance in changing environmental conditions. We constrain our training to only be able to modify the pole lengths the agent is learning from. 
Our goal here is to study the impact of pole length training sequencing in the setup and the replay buffer. The two main ideas are: researching different source task sequencinug and targeted sampling of the replay buffer based on performance. The latter is implicitly linked to pole length performance. To achieve the former we implement two different learning curriculums. Curriculum Learning (CL) is a form of transfer learning where an agent is training on a source task and transitions this learned experience to a target task. CL is the automatic sequencing of these source tasks to improve learning speed and final performance {narvekar2016source} {narvekar2017autonomous}. Given the domain of our problem we select a subset of learning tasks to achieve a generalized agent. The first strategy utilizes an Adaptive Curriculum Learning (ACL) schema that prioritizes weakest performing pole lengths in the training sequencing. The second strategy utilizes an Exploration Diversity Curriculum (EDC) that prioritizes least visited pole lengths in the training sequencing. Lastly, we implement a Prioritized Experience Replay (PER) strategy to sample past transitions (experiences) where a greater Temporal Difference (TD) error was committed {schaul2015prioritized}. Pole length episodes where large errors are committed in the transitions are sampled with greater probability. This implicitly focuses on weak performing pole lengths. 

We investigate the following research questions: 

# Method

Methods: explain your different approaches and its motivation, parameter set-
tings, experimental setup , etc. Make sure everything is reproducible with the
information presented.

ACL 
To promote generalization across varying environmental configurations, we employ an ACL framework. The ACL algorithm uses an object to store pole length rewards, performances, difficulty scores, and a probability distribution for sampling. We set a training cap at t=1500 where t is the 1500th episode. Up until t the agent uses the baseline method of pole length sampling which is where pole lengths are sampled from a discrete uniform distribution. After then t-th episode the agent starts prioriziting the worst performing pole lengths. Performance is judged based on mean reward per episode. We use a look back window parametre LOOK_BACK_WINDOW = 20. This ensures we only consider the recent rewards when computing the performance of a pole length and avoids over stabilizing of performances in our sampling. The ACL algorithm also has an adaptive probability p=0.9 of sampling uniformly from the discrete distribution of pole lengths. 

EDC


PER


General Training Setup





Curriculum learning falls in the domain of transfer learning. Trabnsfer learning is the learning of an agent on a source task to then transition the learned policy, reward function, or model  in the target task. Curriculum learning is the automatic sequencding of source tasks in such a way that an agent can have better performance on a target task; here increased learning speed and performance is the goal {narvekar2016source} {narvekar2017autonomous}. Given the domain of the problem, a subset of learning tasks (source tasks) can be defined for the agent to train on. Randomized curriculum learning, i.e. a curriculum where source tasks are assigned randomly without any given heuristic can delay convergence by adding redundant tasks in the learning sequence {narvekar2017autonomous}. 

The goal we aim to achieve is to study the impact of training with different pole length sequences and policy performance on weakest poel lengths. 
We will research the effect of pole length sequencing 



our research questions + hypotheses




## PDF OUTSIDE ZIP MAYBE? 


# literature


**curriculum learning** 
Curriculum learning falls in the domain of transfer learning. Trabnsfer learning is the learning of an agent on a source task to then transition the learned policy, reward function, or model  in the target task. Curriculum learning is the automatic sequencding of source tasks in such a way that an agent can have better performance on a target task; here increased learning speed and performance is the goal {narvekar2016source} {narvekar2017autonomous}. Given the domain of the problem, a subset of learning tasks (source tasks) can be defined for the agent to train on. Randomized curriculum learning, i.e. a curriculum where source tasks are assigned randomly without any given heuristic can delay convergence by adding redundant tasks in the learning sequence {narvekar2017autonomous}. 

**Prioritized Experience Replay** 
Experience replayes allows an agent to reuse rare and useful prior experiences instead of learning from a given state transition once {lin1992self}. An experience is often denoted as a quadruple (st,a,r,s'). The agent selects an action a in a current state s for which it obtains reward r and transitions to the next state s'. Prioritized experience replay emphasies this further, by selecting which experiences to learn from based on how valuable that experience is in the context of the task. This extended method of experience replay can speed up the learning process of the agent (by a factor of 2) {schaul2015prioritized}. 
We implement prioritized experience replay in ... 

# The Setup 
We deploy a Deep Q Network (DQN) that uses e-learning to tackle the problem of a generalizable cartpole agent. The states in the cartpole env consist of the cart velocity, the cart position (x), the pole angle, and the pole angular velocity. At each time step the agent can choose to move left or right (2-dim aqction space). DQN is used in tasks where the state/action space is too vast to attemtp tabular learning. Q-learning is a suitable approach in the cartpole learning task. DQN utilizes a replay buffer to learn from past and potentially rare experiences. 
The goal is to train an agent such that it can perform well in altering conditions. The conditions that are altered are the pole lengths, wind attacks, and increased horizontal push forces. We aim to achieve optimal performance in these conditions by only altering the pole lkengths that our agent will train with. 

- insert the algorithm  for DQN (loss, update functions, etc.)
- add cartpole env specificities, concise
- add replay buffer
- ADD PARAMETRES AND REASONING FOR THEIR VALUES
per_params are based off of the Schaul et al., 2016  paper {schaul2015prioritized}
batch size params from a 2017 study on optimal batch size tested in the cartpole env {choi2017empirical}
gamma values are taken from the original dqn model paper (atari deep mind paper) {mnih2015human}

# HERE ARE THE TRAINING PARAMS FOR OUR MODELS 
training parametres for the sub 1000 performances (the stable ones):
- epsilon = 0.1
- no decay
- 2000 training steps
- batch size 64
- gamma=0.99 
- alpha=1e-3
- uniform training cap = 1500
- update weights every 4 actions
- update target net every 1000 actions (actions is the count variable incrementing)
- 30 pole lengths
- 50k buffer capacity
- 


# MAIN IDEA (HOW DO WE TRAIN ON POEL LENGTHS ???, ALL RQS TIE INTO THIS MAKE THE STORY FLOW)
- research objective needs to be clear, i.e. exploring 
- 



# Algorithms (strategies)

**Stratified replay buffer** 
We implement a stratified replay buffer as an extension of the replay buffer. This is an implementation of PER. We save experiences as quintuples where we now also store the pole length associated with an experience. After accumulating sufficient experiences to sample from the replay buffer, we ensure that we sample evenly from all experiences. The idea here is that uniform sampling evenly from experiences for every experienced pole length up until a given training episode will ensure the agent is continuously learning from experiences at each altered training condition. The main alteratino we made to our replay buffer was the sampling. When we sample from the replay buffer we compute the samples per pole length we want to sample. If insufficient samples are available, we sample the whole sub buffer that belongs to a given pole length. The sampling within each sub buffer is done using a uniform distribution. 

- add the pseudocode? or concise description
- add our specific sample format
- should the sampling have a refill also (tbd, currently we do not refill so we might be sampling under the limit quite often)

**Adaptive Performance Curriculum**
The adaptive performance curriculum is our first iteration of the Curriculum learning approach. As the agent has to be generalizable, it follows that accumulating experiences in all environment configuratoins will allow the agent to learn from a generalized set of experiences. In this implementation we sequence our source tasks based on performance. The worst performing pole lengths are prioritized in the sequence of learning, such that the agent can attack its weak spots actively as it learns the task. We specify a performance metric for each episode trained on a certain length, so that we can store for each pole length how we've performed so far. 

- performance metric formula
- difficulty score
- prob_distr

**Adatpive Frequency Curriculum**

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

@inproceedings{choi2017empirical,
  title={An empirical study on the optimal batch size for the deep q-network},
  author={Choi, Minsuk},
  booktitle={International Conference on Robot Intelligence Technology and Applications},
  pages={73--81},
  year={2017},
  organization={Springer}
}

@article{mnih2015human,
  title={Human-level control through deep reinforcement learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A and Veness, Joel and Bellemare, Marc G and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K and Ostrovski, Georg and others},
  journal={nature},
  volume={518},
  number={7540},
  pages={529--533},
  year={2015},
  publisher={Nature Publishing Group}
}
