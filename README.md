# AlphaGo: Policy Networks, Value Networks and Tree Search from Scratch in PyTorch

Mastering the game of Go with deep neural networks and tree search (Silver et al., Nature, 2016), rebuilt in PyTorch on a 5 x 5 Go board. No KGS game records are available, so the project gives you a small Go engine and a fixed Go player written in code, and its games stand in for the human expert games. You build the paper's whole pipeline on top: the input feature planes (stone colour, turns since, liberties, capture size, self-atari size, liberties after the move, sensibleness), the 8 board symmetries as data augmentation, the SL policy network trained to predict the expert's moves, the fast linear-softmax rollout policy over local 3 x 3 patterns, the RL policy network improved by REINFORCE self-play against a pool of its earlier versions, the value network trained on one position from each self-play game, and the paper's search: PUCT selection with the SL policy as the prior, separate value-network and rollout statistics with virtual loss, the leaf evaluation mixed by lambda, and the most-visited move at the root. The final script prints our numbers next to the paper's: SL and rollout move accuracy, the RL policy's win rate against the SL policy, the value network's train and test error with one position per game and with every position of each game, and a small tournament of the components and of lambda = 0, 0.5 and 1.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** feature_planes
- [x] **2.** symmetries
- [x] **3.** expert_positions
- [x] **4.** PolicyNet
- [x] **5.** sl_step
- [x] **6.** RolloutPolicy
- [x] **7.** sample_moves
- [x] **8.** self_play
- [x] **9.** train_rl_policy
- [x] **10.** value_positions
- [x] **11.** ValueNet
- [x] **12.** Node
- [x] **13.** backup
- [x] **14.** mcts_move

---

Built on Deep-ML.
