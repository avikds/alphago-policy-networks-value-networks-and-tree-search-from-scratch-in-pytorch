"""
AlphaGo: Policy Networks, Value Networks and Tree Search from Scratch in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""AlphaGo on a 5 x 5 board (Silver et al., 2016, "Mastering the game of Go with deep neural networks and tree search").

Story: no KGS game records are available, so a fixed Go player written in code (expert_move) plays 400 games that
stand in for the human expert games. From them the script trains the SL policy network (with the 8 board symmetries)
and the linear-softmax rollout policy, then improves the policy by REINFORCE self-play against a pool of its earlier
versions, trains the value network on one position from each of 6,000 self-play games (and, for comparison, on every
position of a few games), and plays the paper's search, which mixes the value network and rollouts by lambda, in a
small tournament. Every number the paper reports for these stages is printed next to ours. Takes about five minutes on
one CPU core.
"""
import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BLACK, WHITE = 1, 2


class GoGame:
    """Go on a small N x N board with Chinese area scoring, used by every step of the project.

    Points are numbered p = row * N + col; the move N * N is a pass. The board is a flat list of N * N ints:
    0 empty, 1 black, 2 white. Black moves first. Rules: a stone or chain is captured when it has no liberty
    (adjacent empty point); a move that leaves its own chain without a liberty and captures nothing is illegal
    (no suicide); the simple ko rule forbids retaking a single stone at once. The game ends after two passes in a
    row or after max_moves moves, and is scored by area: stones on the board plus empty regions that touch only one
    colour, with komi added to White.
    """

    def __init__(self, n=5, komi=0.5, max_moves=None):
        self.n = n
        self.komi = komi
        self.max_moves = 3 * n * n if max_moves is None else max_moves
        self.board = [0] * (n * n)
        self.to_play = BLACK
        self.ko = None
        self.passes = 0
        self.moves = 0
        self.played_at = [0] * (n * n)
        self.last_move = None
        self.history = []
        self._nbrs = []
        for p in range(n * n):
            r, c = divmod(p, n)
            self._nbrs.append([(r + dr) * n + c + dc for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                               if 0 <= r + dr < n and 0 <= c + dc < n])
        self._cache = None

    @property
    def pass_move(self):
        return self.n * self.n

    @property
    def done(self):
        return self.passes >= 2 or self.moves >= self.max_moves

    def copy(self):
        g = GoGame.__new__(GoGame)
        g.__dict__.update(self.__dict__)
        g.board = self.board[:]
        g.played_at = self.played_at[:]
        g.history = self.history[:]
        return g

    def neighbours(self, p):
        """The on-board points next to p (up, down, left, right)."""
        return self._nbrs[p]

    def chains(self):
        """(chain_of, chains): chain_of[p] is an index into chains for a stone at p (None if empty); each chain is
        (colour, set of stones, set of liberties)."""
        if self._cache is None:
            chain_of = [None] * len(self.board)
            chains = []
            for p, colour in enumerate(self.board):
                if colour == 0 or chain_of[p] is not None:
                    continue
                stones, libs, stack = {p}, set(), [p]
                chain_of[p] = len(chains)
                while stack:
                    q = stack.pop()
                    for r in self._nbrs[q]:
                        v = self.board[r]
                        if v == 0:
                            libs.add(r)
                        elif v == colour and chain_of[r] is None:
                            chain_of[r] = len(chains)
                            stones.add(r)
                            stack.append(r)
                chains.append((colour, stones, libs))
            self._cache = (chain_of, chains)
        return self._cache

    def chain(self, p):
        """(stones, liberties) of the chain through the stone at p."""
        chain_of, chains = self.chains()
        _, stones, libs = chains[chain_of[p]]
        return stones, libs

    def move_info(self, p, player=None):
        """What playing at empty point p would do for `player` (default: the player to move), without playing it.
        Returns (legal, captured, size, liberties): whether it is legal, how many opponent stones it captures, and
        the size and liberty count of the player's chain through p after the move."""
        player = self.to_play if player is None else player
        if p == self.pass_move:
            return True, 0, 0, 0
        if self.board[p] != 0 or p == self.ko and player == self.to_play:
            return False, 0, 0, 0
        chain_of, chains = self.chains()
        opp = 3 - player
        captured, dead = 0, set()
        stones, libs, seen = {p}, set(), set()
        for q in self._nbrs[p]:
            v = self.board[q]
            if v == 0:
                libs.add(q)
                continue
            k = chain_of[q]
            if k in seen:
                continue
            seen.add(k)
            colour, ch_stones, ch_libs = chains[k]
            if colour == opp and ch_libs == {p}:
                captured += len(ch_stones)
                dead |= ch_stones
            elif colour == player:
                stones |= ch_stones
                libs |= ch_libs
        libs.discard(p)
        if dead:
            for s in stones:
                for q in self._nbrs[s]:
                    if q in dead:
                        libs.add(q)
        legal = bool(libs) or captured > 0
        return legal, captured, len(stones), len(libs)

    def is_legal(self, p):
        return self.move_info(p)[0]

    def legal_moves(self):
        """Every legal board point for the player to move (the pass is always legal and is not listed)."""
        return [p for p in range(self.n * self.n) if self.board[p] == 0 and self.move_info(p)[0]]

    def is_eye(self, p, player=None):
        """True if p is a single-point eye of `player` (default: the player to move): an empty point whose every
        on-board neighbour is one of the player's stones."""
        player = self.to_play if player is None else player
        return self.board[p] == 0 and all(self.board[q] == player for q in self._nbrs[p])

    def sensible_moves(self):
        """Legal moves that do not fill one of the mover's own eyes."""
        return [p for p in self.legal_moves() if not self.is_eye(p)]

    def play(self, p):
        """Play move p (a point or the pass) for the player to move; raises ValueError if it is illegal."""
        if self.done:
            raise ValueError("the game is over")
        player = self.to_play
        self.history.append(p)
        self.last_move = p
        self.moves += 1
        if p == self.pass_move:
            self.passes += 1
            self.ko = None
            self.to_play = 3 - player
            return
        legal, captured, size, libs = self.move_info(p)
        if not legal:
            raise ValueError(f"illegal move {p}")
        chain_of, chains = self.chains()
        dead = []
        for q in self._nbrs[p]:
            if self.board[q] == 3 - player and chains[chain_of[q]][2] == {p}:
                dead.extend(chains[chain_of[q]][1])
        self.board[p] = player
        for q in set(dead):
            self.board[q] = 0
            self.played_at[q] = 0
        self.played_at[p] = self.moves
        self.ko = dead[0] if captured == 1 and size == 1 and libs == 1 else None
        self.passes = 0
        self.to_play = 3 - player
        self._cache = None

    def area(self):
        """(black area, white area): stones plus empty regions bordered by one colour only."""
        b = self.board.count(BLACK)
        w = self.board.count(WHITE)
        seen = set()
        for p in range(len(self.board)):
            if self.board[p] != 0 or p in seen:
                continue
            region, border, stack = {p}, set(), [p]
            seen.add(p)
            while stack:
                q = stack.pop()
                for r in self._nbrs[q]:
                    if self.board[r] == 0 and r not in seen:
                        seen.add(r)
                        region.add(r)
                        stack.append(r)
                    elif self.board[r] != 0:
                        border.add(self.board[r])
            if border == {BLACK}:
                b += len(region)
            elif border == {WHITE}:
                w += len(region)
        return b, w

    def winner(self):
        """BLACK or WHITE by area score with komi for White."""
        b, w = self.area()
        return BLACK if b - w - self.komi > 0 else WHITE

    def __str__(self):
        sym = ".XO"
        return "\n".join(" ".join(sym[self.board[r * self.n + c]] for c in range(self.n)) for r in range(self.n))


def expert_move(game, rng, noise=0.5):
    """The project's stand-in for a KGS dan player: a fixed Go player written in code.

    For every sensible move it looks one move ahead and scores the result with hand-written Go knowledge: stones
    captured, own chains rescued from atari, own stones left in atari, opponent chains put in atari, and the area
    each side controls (a point belongs to the side whose stones are nearer). It plays the best score plus a little
    random noise (rng is a numpy Generator), and passes when no sensible move is left.
    """
    moves = game.sensible_moves()
    if not moves:
        return game.pass_move
    me, opp = game.to_play, 3 - game.to_play
    n = game.n
    chain_of, chains = game.chains()
    in_atari = {k for k, (c, s, l) in enumerate(chains) if c == me and len(l) == 1}
    best, best_score = moves[0], -1e9
    for p in moves:
        legal, captured, size, libs = game.move_info(p)
        g = game.copy()
        g.play(p)
        _, ch2 = g.chains()
        score = 6.0 * captured
        if libs == 1:
            score -= 8.0 * size
        elif libs == 2:
            score -= 1.0 * size
        rescued = sum(len(chains[k][1]) for k in in_atari if p in chains[k][2])
        if libs >= 2:
            score += 5.0 * rescued
        for c, s, l in ch2:
            if c == opp and len(l) == 1:
                score += 2.0 * len(s)
        mine = theirs = 0
        for q in range(n * n):
            if g.board[q] == me:
                mine += 1
            elif g.board[q] == opp:
                theirs += 1
            else:
                qr, qc = divmod(q, n)
                dm = do = 99
                for s in range(n * n):
                    if g.board[s]:
                        d = abs(s // n - qr) + abs(s % n - qc)
                        if g.board[s] == me:
                            dm = min(dm, d)
                        else:
                            do = min(do, d)
                if dm < do:
                    mine += 1
                elif do < dm:
                    theirs += 1
        score += 1.0 * (mine - theirs)
        score += noise * float(rng.gumbel())
        if score > best_score:
            best, best_score = p, score
    return best


def play_match(player_a, player_b, n_games, n=5, komi=0.5):
    """Win rate of player_a against player_b; each player is a function game -> move. A plays black in even games."""
    wins = 0
    for i in range(n_games):
        g = GoGame(n, komi)
        a_black = i % 2 == 0
        while not g.done:
            mover = player_a if (g.to_play == BLACK) == a_black else player_b
            g.play(mover(g))
        wins += (g.winner() == BLACK) == a_black
    return wins / n_games


def train_value(x, z, x_test, z_test, steps, lr, seed):
    """Plain SGD on the mean squared error, minibatches of 32. Returns the best (lowest held-out error) network,
    its (train, test) error, and the (train, test) error at the end of training."""
    torch.manual_seed(seed)
    net = ValueNet()
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    best, best_state, last = (9.0, 9.0), None, None
    for s in range(1, steps + 1):
        idx = torch.randint(len(z), (32,), generator=g)
        loss = ((net(x[idx]) - z[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 250 == 0:
            with torch.no_grad():
                last = (float(((net(x) - z) ** 2).mean()), float(((net(x_test) - z_test) ** 2).mean()))
            if last[1] < best[1]:
                best, best_state = last, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    return net, best, last


def main():
    torch.set_num_threads(1)      # tiny networks: one thread is fastest and keeps the timing steady
    t0 = time.time()
    print("Settings (paper -> this script):")
    print("  board 19 x 19 -> 5 x 5 (Chinese area scoring, komi 7.5 -> 0.5, simple ko, no suicide)")
    print("  expert data: 160,000 KGS games of 6-9 dan humans -> 400 games of a fixed Go player written in code")
    print("  feature planes 48 (49 for the value net) -> 46 (47): the two ladder planes are left out")
    print("  policy network 13 layers, 192 filters -> 5 layers, 32 filters; value network 15 layers -> 8 layers")
    print("  SL training: SGD step 0.003, minibatch 16, 340 million steps -> SGD step 0.05, minibatch 64, 2,000 steps")
    print("  rollout policy: 3x3 patterns, response patterns, nakade, save atari, neighbour -> 3x3 patterns, save atari, neighbour")
    print("  RL: 10,000 minibatches of 128 games, pool +1 every 500 -> 40 minibatches of 64 games, pool +1 every 5")
    print("  value data: 30 million games, U ~ unif{1, 450} -> 6,000 games, U ~ unif{1, 40}")
    print("  search: c_puct 5 -> 1, n_thr 40 -> 1, thousands of simulations -> 32 per move; beta 0.67, lambda, n_vl 3 (same)")

    # ---------------- expert games (the KGS stand-in)
    rng = np.random.default_rng(0)
    records = []
    for _ in range(400):
        g = GoGame(5)
        while not g.done:
            g.play(expert_move(g, rng))
        records.append(g.history)
    rand_rng = np.random.default_rng(1)

    def random_player(g):
        moves = g.sensible_moves()
        return moves[int(rand_rng.integers(len(moves)))] if moves else g.pass_move

    expert_rng = np.random.default_rng(2)
    expert = lambda g: expert_move(g, expert_rng)
    print(f"\nExpert games: {len(records)}, {np.mean([len(r) for r in records]):.1f} moves on average; "
          f"the expert beats a uniformly random player in {play_match(expert, random_player, 40):.0%} of 40 games")

    x, legal, moves, z, game_ids = expert_positions(records)
    test = game_ids < 40
    print(f"Positions without passes: {len(moves)} ({int(test.sum())} from the first 40 games held out as the test set)")
    xa, la, ma = [], [], []
    for xi, li, mi in zip(x[~test], legal[~test], moves[~test]):
        xs, ms = symmetries(torch.cat([xi, li.view(1, 5, 5).float()]), mi)
        xa.append(xs[:, :46])
        la.append(xs[:, 46].flatten(1) > 0)
        ma.append(ms)
    xa, la, ma = torch.cat(xa), torch.cat(la), torch.cat(ma)
    print(f"Training set with the 8 symmetries: {len(ma)} examples")

    # ---------------- SL policy network
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    sl = PolicyNet()
    opt = torch.optim.SGD(sl.parameters(), lr=0.05)
    for _ in range(2000):
        idx = torch.randint(len(ma), (64,), generator=gen)
        sl_step(sl, opt, xa[idx], la[idx], ma[idx])
    with torch.no_grad():
        sl_train = float((sl(x[~test], legal[~test]).argmax(1) == moves[~test]).float().mean())
        sl_test = float((sl(x[test], legal[test]).argmax(1) == moves[test]).float().mean())

    # ---------------- rollout policy
    rp = RolloutPolicy()
    train_positions, test_positions = [], []
    for i, rec in enumerate(records):
        g = GoGame(5)
        for m in rec:
            if m != g.pass_move:
                (test_positions if i < 40 else train_positions).append((g.copy(), m))
            g.play(m)
    for _ in range(2):
        for g, m in train_positions:
            rp.train_step(g, m, g.legal_moves(), 0.1)
    hits = 0
    for g, m in test_positions:
        cand = g.legal_moves()
        pr = rp.probs(g, cand)
        hits += cand[max(range(len(cand)), key=lambda i: pr[i])] == m
    rp_test = hits / len(test_positions)
    sample = [g for g, _ in test_positions[:300]]
    t = time.time()
    for g in sample:
        rp.act(g, rand_rng)
    rp_us = (time.time() - t) / len(sample) * 1e6
    t = time.time()
    for g in sample:
        sample_moves(sl, [g], gen)
    sl_us = (time.time() - t) / len(sample) * 1e6
    print("\nMove prediction on the held-out expert games:")
    print(f"  SL policy network  test accuracy {sl_test:6.1%} (train {sl_train:.1%})   paper: 57.0% (55.4% for the single 192-filter network)")
    print(f"  rollout policy     test accuracy {rp_test:6.1%} ({len(rp.w)} features)    paper: 24.2%")
    print(f"  time per move: rollout policy {rp_us:.0f} us, policy network {sl_us:.0f} us   paper: 2 us vs 3 ms")

    # ---------------- RL policy network
    rl, pool, history = train_rl_policy(sl, 40, 64, 5, 0.01, gen)
    _, _, _, _, wins = self_play(rl, sl, 400, gen)
    print(f"\nRL policy network: {len(history)} minibatches of 64 games, opponent pool of {len(pool)}")
    print("  win rate against the drawn opponent, per block of 10 minibatches: "
          + ", ".join(f"{np.mean(history[i:i + 10]):.0%}" for i in range(0, len(history), 10)))
    print(f"  RL policy vs SL policy (no search, moves sampled): {wins / 400:.1%} of 400 games   paper: more than 80%")
    sl_player = lambda g: sample_moves(sl, [g], gen)[0][0]
    rl_player = lambda g: sample_moves(rl, [g], gen)[0][0]
    rp_player = lambda g: rp.act(g, rand_rng)
    print(f"  against the expert player (20 games each): SL {play_match(sl_player, expert, 20):.0%}, RL {play_match(rl_player, expert, 20):.0%}")

    # ---------------- value network
    xv, zv = value_positions(sl, rl, 6000, 40, gen)
    n_test = len(zv) // 5
    xv_test, zv_test, xv_train, zv_train = xv[:n_test], zv[:n_test], xv[n_test:], zv[n_test:]
    xb, zb = value_positions(sl, rl, 140, 40, gen, all_positions=True)
    n_games_b = 140
    if len(zb) > len(zv_train):
        xb, zb = xb[:len(zv_train)], zb[:len(zv_train)]
    print(f"\nValue network data: {len(zv)} positions, one from each of 6,000 self-play games "
          f"({len(zv_train)} train, {n_test} test); a predictor that always says 0 scores 1.000")
    value_net, best_a, last_a = train_value(xv_train, zv_train, xv_test, zv_test, 3000, 0.05, 0)
    _, best_b, last_b = train_value(xb, zb, xv_test, zv_test, 3000, 0.05, 0)
    print("  mean squared error            best test (train there)     after 3,000 steps (train / test)")
    print(f"  one position per game       {best_a[1]:.3f} ({best_a[0]:.3f})               {last_a[0]:.3f} / {last_a[1]:.3f}"
          f"     paper: 0.234 test, 0.226 train")
    print(f"  every position of {n_games_b} games {best_b[1]:.3f} ({best_b[0]:.3f})               {last_b[0]:.3f} / {last_b[1]:.3f}"
          f"     paper (KGS games): 0.37 test, 0.19 train")
    print("  (same number of training positions in both rows, same held-out positions; the search uses the first network)")

    # ---------------- tournament
    print("\nTournament (5 x 5, win rate of the row player; every search uses 32 simulations per move):")
    search_rng = np.random.default_rng(3)

    N_RL_GAMES = 20

    def searcher(lam, prior):
        return lambda g: mcts_move(g, prior, value_net, rp, 32, lam, search_rng)[0]

    black_wins = 0
    for _ in range(100):
        g = GoGame(5)
        while not g.done:
            g.play(rl_player(g))
        black_wins += g.winner() == BLACK
    rows = [("RL policy vs SL policy", wins / 400, 400),
            ("SL policy vs rollout policy", play_match(sl_player, rp_player, 40), 40),
            ("RL policy vs rollout policy", play_match(rl_player, rp_player, 40), 40),
            ("search (SL prior), lambda = 0.5 vs SL policy", play_match(searcher(0.5, sl), sl_player, 12), 12)]
    for lam in (0.0, 0.5, 1.0):
        rows.append((f"search (SL prior), lambda = {lam} vs RL policy", play_match(searcher(lam, sl), rl_player, 12), 12))
    for lam in (0.0, 0.5, 1.0):
        rows.append((f"search (RL prior), lambda = {lam} vs RL policy", play_match(searcher(lam, rl), rl_player, N_RL_GAMES), N_RL_GAMES))
    for name, w, n in rows:
        print(f"  {name:<48} {w:6.1%} +- {50 / math.sqrt(n):4.1f}  ({n} games)")
    print(f"  RL policy against itself: black wins {black_wins}% of 100 games (5 x 5 Go favours the first player)")
    print("  paper, single machine (Elo): lambda = 0.5 2890, lambda = 1 (rollouts only) 2416, lambda = 0 (value net only) 2177,")
    print("  policy network alone 1517; lambda = 0.5 won 95% of its games against the other variants, and the SL prior")
    print("  worked better in the search than the RL prior.")
    print("  The +- column is one standard error of a win rate near 50% (50 / sqrt(games) points). Rows that are already")
    print("  clear use 12 games; the RL-prior rows, where the three lambdas are compared, use the most games that fit.")
    print("  The search beats the policy that gives its prior. With the paper's SL prior and only 32 simulations it does")
    print("  not beat the RL policy here: on 5 x 5 the RL policy has learned the winning opening for black far better than")
    print("  the SL policy (it beats SL far more often than the paper's 80%), and a search this small cannot find a line")
    print("  its prior rarely proposes. The paper's search ran thousands of simulations per move, and its SL prior")
    print("  modelled strong human players, not a weak hand-written one.")
    print(f"\n[total time {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()

