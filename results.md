```
Settings (paper -> this script):
  board 19 x 19 -> 5 x 5 (Chinese area scoring, komi 7.5 -> 0.5, simple ko, no suicide)
  expert data: 160,000 KGS games of 6-9 dan humans -> 400 games of a fixed Go player written in code
  feature planes 48 (49 for the value net) -> 46 (47): the two ladder planes are left out
  policy network 13 layers, 192 filters -> 5 layers, 32 filters; value network 15 layers -> 8 layers
  SL training: SGD step 0.003, minibatch 16, 340 million steps -> SGD step 0.05, minibatch 64, 2,000 steps
  rollout policy: 3x3 patterns, response patterns, nakade, save atari, neighbour -> 3x3 patterns, save atari, neighbour
  RL: 10,000 minibatches of 128 games, pool +1 every 500 -> 40 minibatches of 64 games, pool +1 every 5
  value data: 30 million games, U ~ unif{1, 450} -> 6,000 games, U ~ unif{1, 40}
  search: c_puct 5 -> 1, n_thr 40 -> 1, thousands of simulations -> 32 per move; beta 0.67, lambda, n_vl 3 (same)

Expert games: 400, 42.9 moves on average; the expert beats a uniformly random player in 95% of 40 games
Positions without passes: 15405 (1452 from the first 40 games held out as the test set)
Training set with the 8 symmetries: 111624 examples

Move prediction on the held-out expert games:
  SL policy network  test accuracy  71.4% (train 72.5%)   paper: 57.0% (55.4% for the single 192-filter network)
  rollout policy     test accuracy  51.2% (12527 features)    paper: 24.2%
  time per move: rollout policy 163 us, policy network 1108 us   paper: 2 us vs 3 ms

RL policy network: 40 minibatches of 64 games, opponent pool of 9
  win rate against the drawn opponent, per block of 10 minibatches: 74%, 81%, 74%, 64%
  RL policy vs SL policy (no search, moves sampled): 90.8% of 400 games   paper: more than 80%
  against the expert player (20 games each): SL 40%, RL 75%

Value network data: 5651 positions, one from each of 6,000 self-play games (4521 train, 1130 test); a predictor that always says 0 scores 1.000
  mean squared error            best test (train there)     after 3,000 steps (train / test)
  one position per game       0.526 (0.428)               0.370 / 0.573     paper: 0.234 test, 0.226 train
  every position of 140 games 0.646 (0.488)               0.166 / 0.802     paper (KGS games): 0.37 test, 0.19 train
  (same number of training positions in both rows, same held-out positions; the search uses the first network)

Tournament (5 x 5, win rate of the row player; every search uses 32 simulations per move):
  RL policy vs SL policy                            90.8% +-  2.5  (400 games)
  SL policy vs rollout policy                       80.0% +-  7.9  (40 games)
  RL policy vs rollout policy                       95.0% +-  7.9  (40 games)
  search (SL prior), lambda = 0.5 vs SL policy      91.7% +- 14.4  (12 games)
  search (SL prior), lambda = 0.0 vs RL policy       0.0% +- 14.4  (12 games)
  search (SL prior), lambda = 0.5 vs RL policy       8.3% +- 14.4  (12 games)
  search (SL prior), lambda = 1.0 vs RL policy      16.7% +- 14.4  (12 games)
  search (RL prior), lambda = 0.0 vs RL policy      40.0% +- 11.2  (20 games)
  search (RL prior), lambda = 0.5 vs RL policy      60.0% +- 11.2  (20 games)
  search (RL prior), lambda = 1.0 vs RL policy      55.0% +- 11.2  (20 games)
  RL policy against itself: black wins 63% of 100 games (5 x 5 Go favours the first player)
  paper, single machine (Elo): lambda = 0.5 2890, lambda = 1 (rollouts only) 2416, lambda = 0 (value net only) 2177,
  policy network alone 1517; lambda = 0.5 won 95% of its games against the other variants, and the SL prior
  worked better in the search than the RL prior.
  The +- column is one standard error of a win rate near 50% (50 / sqrt(games) points). Rows that are already
  clear use 12 games; the RL-prior rows, where the three lambdas are compared, use the most games that fit.
  The search beats the policy that gives its prior. With the paper's SL prior and only 32 simulations it does
  not beat the RL policy here: on 5 x 5 the RL policy has learned the winning opening for black far better than
  the SL policy (it beats SL far more often than the paper's 80%), and a search this small cannot find a line
  its prior rarely proposes. The paper's search ran thousands of simulations per move, and its SL prior
  modelled strong human players, not a weak hand-written one.

[total time 319s]
```
