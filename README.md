# YGO-BOT: Distributed Deep Reinforcement Learning for Yu-Gi-Oh!

YGO-BOT is an open-source, massive-scale Deep Reinforcement Learning project aimed at creating a super-human AI for the Yu-Gi-Oh! Trading Card Game. 
It combines **Proximal Policy Optimization (PPO)**, **Gumbel AlphaZero MCTS**, and a fully distributed **Ray** architecture, all powered by **JAX/Flax** for maximum vectorization and hardware utilization.

This project is currently designed to scale on massive cloud computing infrastructures to tackle one of the most complex board/card games in existence.

---

## 🧠 The Challenge: Why Yu-Gi-Oh! ?

Yu-Gi-Oh! presents unique challenges for Artificial Intelligence, far exceeding the complexity of Chess or Go:
- **Massive State Space**: Over 10,000 unique cards with complex, chaining effects. Our observation vector has **60,694 dimensions**.
- **Hidden Information**: Hands and face-down cards are hidden from the opponent (Imperfect Information game).
- **Complex Action Space**: Up to 200 contextual actions (Summon, Activate, Chain, Attack) that dynamically change based on game state.

---

## 🏗️ Technical Architecture

### 1. Neural Network (JAX/Flax)
The core of the agent is an `ActorCriticLSTM` network written purely in **Flax/JAX**:
- **Embeddings Layer**: Projects categorical card IDs into a dense continuous space.
- **LSTM Core**: Maintains a hidden state (`carry_state`) across turns to deal with partial observability and long-term planning.
- **Action Masking**: A critical mechanism that zeroes out illegal actions dynamically at the logits level, preventing the network from wasting gradient updates on invalid moves.

### 2. Distributed Asynchronous Self-Play (Ray)
To overcome the massive computational requirement of environment simulation, we implemented a decoupled actor-learner architecture using **Ray**:
- **Rollout Workers (CPU)**: Hundreds of lightweight actors run the C++ Yu-Gi-Oh! engine (`libocgcore`) to simulate games in parallel. They are restricted from GPU access to avoid VRAM bottlenecks.
- **Learner (GPU/TPU)**: A centralized actor that receives batches of rollouts (States, Actions, Rewards, Advantages) and performs rapid Backpropagation and Policy Optimization using JAX's `jit` compilation.
- **Parameter Server (SelfPlayManager)**: Maintains a history of network weights. Workers regularly pull older snapshots to train the agent against previous versions of itself, ensuring robust monotonic improvement.

### 3. Gumbel AlphaZero & PPO
We completely bypass manual *Reward Shaping* (which leads to Reward Hacking) and rely purely on the sparse end-game reward (+1 Win, -1 Loss). To achieve this:
- **MCTS with Gumbel Noise**: We implement a Monte Carlo Tree Search at the rollout level. By injecting **Gumbel noise** at the root (Gumbel AlphaZero variant), the agent achieves stable exploration without needing hundreds of simulations.
- **Curriculum Learning**: The architecture supports toggling MCTS off for a "Cold Start" rapid pure-PPO training, then activating MCTS for deep tactical fine-tuning.
- **Generalized Advantage Estimation (GAE)**: Used to stabilize the policy updates over extremely long episodes.

---

## 🚀 Why We Need Google TRC (TPU Research Cloud)

While the architecture is mathematically sound and highly optimized, Yu-Gi-Oh! requires an enormous amount of self-play to converge.
- A single Rollout Worker needs to clone the C++ environment state hundreds of times per game to build the MCTS tree.
- On a local machine (RTX 3070 Ti + 8-core CPU), generating a single batch of rollouts can take several minutes.
- To reach professional human level, the agent must play **millions of games**.

**Google TPUs** (via JAX's native XLA compilation) combined with a massive CPU cluster for Ray workers is the only way to scale this project from an architectural MVP to a superhuman agent. The code is already TPU-ready through JAX.

---

## 🛠️ Setup & Local Training

### Requirements
- Python 3.12+
- WSL2 (for Windows users, required for the C++ engine)
- JAX & Flax
- Ray

### Running the Distributed Training
To launch the cluster locally and begin the PPO Self-Play loop:
```bash
python scripts/train_distributed.py
```
*(Tip: Set `PYTHONUNBUFFERED=1` to see real-time Ray logs)*

---

*This project is built upon `libocgcore` and draws inspiration from AlphaZero and deep reinforcement learning breakthroughs.*
