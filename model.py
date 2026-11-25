"""
AlphaGo: Policy Networks, Value Networks and Tree Search from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - feature_planes
import torch

def feature_planes(game, value=False):
    """
    Construct the AlphaGo input feature planes for a Go position.

    Returns:
        torch.float32 tensor of shape (46, n, n), or (47, n, n)
        when value=True.

    The feature planes are represented relative to the player to move.
    """

    n = game.n
    num_planes = 47 if value else 46

    # Initialize all feature planes to zero.
    planes = torch.zeros((num_planes, n, n), dtype=torch.float32)

    player = game.to_play
    opponent = 3 - player

    # Convert a count or collection into an integer count.
    # This supports both scalar counts and collections returned by the Go engine.
    def count_items(items):
        if items is None:
            return 0

        try:
            return len(items)
        except TypeError:
            return int(items)

    # ---------------------------------------------------------
    # Planes 0-2: Stone colours and empty points
    # ---------------------------------------------------------

    for p in range(n * n):
        r, c = divmod(p, n)
        stone = game.board[p]

        if stone == player:
            planes[0, r, c] = 1.0

        elif stone == opponent:
            planes[1, r, c] = 1.0

        else:
            # Empty intersection.
            planes[2, r, c] = 1.0

    # ---------------------------------------------------------
    # Plane 3: Constant plane of ones
    # ---------------------------------------------------------

    planes[3].fill_(1.0)

    # ---------------------------------------------------------
    # Planes 4-11: Turns since each stone was placed
    # Planes 12-19: Liberties of each stone's chain
    # ---------------------------------------------------------

    for p in range(n * n):

        if game.board[p] == 0:
            continue

        r, c = divmod(p, n)

        # Number of moves since this stone was placed.
        turns_since = game.moves - game.played_at[p] + 1
        turns_since = max(1, min(turns_since, 8))

        # One-hot encoding of turns since placement.
        planes[4 + turns_since - 1, r, c] = 1.0

        # Obtain the chain and its liberties.
        stones, liberties = game.chain(p)

        liberty_count = count_items(liberties)
        liberty_count = max(1, min(liberty_count, 8))

        # One-hot encoding of the chain's liberty count.
        planes[12 + liberty_count - 1, r, c] = 1.0

    # ---------------------------------------------------------
    # Planes 20-27: Capture size
    # Planes 28-35: Self-atari size
    # Planes 36-43: Liberties after the move
    # Plane 44: Sensibleness
    #
    # These features are computed only for legal moves on
    # currently empty intersections.
    # ---------------------------------------------------------

    for p in range(n * n):

        if game.board[p] != 0:
            continue

        # Get move information from the Go engine.
        legal, captured, size, liberties = game.move_info(p)

        # Illegal moves do not activate any move-related features.
        if not legal:
            continue

        r, c = divmod(p, n)

        capture_count = count_items(captured)
        chain_size = count_items(size)
        liberty_count = count_items(liberties)

        # -----------------------------------------------------
        # Planes 20-27: Capture size
        # Activate the corresponding plane only when the move
        # captures at least one opponent stone.
        # -----------------------------------------------------

        if capture_count > 0:
            capture_bucket = min(capture_count, 8)
            planes[20 + capture_bucket - 1, r, c] = 1.0

        # -----------------------------------------------------
        # Planes 28-35: Self-atari size
        # A move is self-atari when its resulting chain has
        # exactly one liberty.
        # -----------------------------------------------------

        if liberty_count == 1:
            self_atari_bucket = min(max(chain_size, 1), 8)
            planes[28 + self_atari_bucket - 1, r, c] = 1.0

        # -----------------------------------------------------
        # Planes 36-43: Liberties after the move
        # One-hot encoding of the resulting chain's liberties.
        # -----------------------------------------------------

        liberty_bucket = min(max(liberty_count, 1), 8)
        planes[36 + liberty_bucket - 1, r, c] = 1.0

        # -----------------------------------------------------
        # Plane 44: Sensibleness
        # A move is sensible if it does not fill the player's
        # own eye.
        # -----------------------------------------------------

        if not game.is_eye(p):
            planes[44, r, c] = 1.0

    # ---------------------------------------------------------
    # Plane 45: Reserved zero plane
    # It remains zero as initialized above.
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Plane 46: Player colour for value-network training
    # One if Black is to play; otherwise zero.
    # Included only when value=True.
    # ---------------------------------------------------------

    if value and player == 1:
        planes[46].fill_(1.0)

    return planes

# Step 2 - symmetries
def symmetries(planes, move):
    """
    Generate all 8 rotations and reflections of a Go training example.

    Args:
        planes: Tensor of shape (C, n, n).
        move: Integer representing a point as row * n + col.

    Returns:
        xs: Tensor of shape (8, C, n, n), containing the transformed
            feature planes. Index 0 is the original example.
        moves: int64 tensor of shape (8,), containing the corresponding
               transformed move positions.
    """

    n = planes.shape[-1]

    # Convert the flattened move index into row and column coordinates.
    row, col = divmod(int(move), n)

    # Create a one-hot board marking the original move position.
    # Using the same device and dtype as the feature planes ensures
    # compatibility with CPU and GPU tensors.
    move_mask = torch.zeros(
        (n, n),
        dtype=planes.dtype,
        device=planes.device
    )
    move_mask[row, col] = 1.0

    transformed_planes = []
    transformed_moves = []

    # Generate four rotations and their four horizontal reflections.
    for j in range(8):

        # Apply j % 4 counterclockwise quarter-turn rotations.
        rotated_planes = torch.rot90(
            planes,
            k=j % 4,
            dims=(-2, -1)
        )

        rotated_move = torch.rot90(
            move_mask,
            k=j % 4,
            dims=(-2, -1)
        )

        # For indices 4-7, additionally reflect the board
        # from left to right.
        if j >= 4:
            rotated_planes = torch.flip(
                rotated_planes,
                dims=(-1,)
            )

            rotated_move = torch.flip(
                rotated_move,
                dims=(-1,)
            )

        # Store the transformed feature planes.
        transformed_planes.append(rotated_planes)

        # Find the flattened position of the transformed move.
        # argmax returns its index as a torch.int64 scalar.
        new_move = torch.argmax(rotated_move.reshape(-1))

        transformed_moves.append(new_move)

    # Stack all eight transformations along a new leading dimension.
    xs = torch.stack(transformed_planes, dim=0)
    moves = torch.stack(transformed_moves).to(torch.int64)

    return xs, moves

# Step 3 - expert_positions
def expert_positions(records, n=5, komi=0.5):
    """
    Replay expert Go games and construct supervised training examples.

    Each non-pass move produces one training example containing:
        - Feature planes of the current position.
        - Legal move mask.
        - Move played.
        - Game outcome from the perspective of the player to move.
        - Index of the game in records.

    Args:
        records: List of games, where each game is a list of moves.
                 Moves are represented as row * n + col, with n * n
                 representing a pass.
        n: Go board size.
        komi: Komi used by the Go engine.

    Returns:
        x:        Float32 tensor of shape (M, 46, n, n).
        legal:    Boolean tensor of shape (M, n*n).
        moves:    Int64 tensor of shape (M,).
        z:        Float32 tensor of shape (M,).
        game_ids: Int64 tensor of shape (M,).
    """

    # Store training examples and their corresponding metadata.
    xs = []
    legals = []
    moves = []
    outcomes = []
    game_ids = []

    # Replay each expert game independently.
    for game_idx, record in enumerate(records):

        # Every game starts from a fresh, empty Go board.
        game = GoGame(n, komi)

        # Keep track of the player who made each recorded move.
        # Outcomes can only be assigned after the game finishes.
        game_players = []

        for move in record:

            move = int(move)

            # Identify pass moves.
            is_pass = (move == n * n)

            # Record a training example before playing each non-pass move.
            if not is_pass:

                # Current player to move.
                player = game.to_play

                # Extract the 46 input feature planes.
                position = feature_planes(game, value=False)

                xs.append(position)

                # Construct a legal move mask.
                # A point must be both empty and legal for the current player.
                legal_mask = torch.tensor(
                    [
                        game.board[p] == 0 and game.is_legal(p)
                        for p in range(n * n)
                    ],
                    dtype=torch.bool
                )

                legals.append(legal_mask)

                # Store the expert's move.
                moves.append(move)

                # Store the player who made this move.
                game_players.append(player)

                # Store the index of the source game.
                game_ids.append(game_idx)

            # Play the move, including passes.
            # Passes advance the game but do not generate training examples.
            game.play(move)

        # ---------------------------------------------------------
        # Assign outcomes after the entire game has been replayed.
        # ---------------------------------------------------------

        winner = game.winner()

        # Outcome is relative to the player who was to move
        # when each training position was recorded.
        for player in game_players:
            if player == winner:
                outcomes.append(1.0)
            else:
                outcomes.append(-1.0)

    # ---------------------------------------------------------
    # Convert collected examples into tensors with the required
    # shapes and data types.
    # ---------------------------------------------------------

    if xs:
        x = torch.stack(xs).to(dtype=torch.float32)
        legal = torch.stack(legals)
    else:
        # Handle an empty dataset while preserving the required shapes.
        x = torch.empty((0, 46, n, n), dtype=torch.float32)
        legal = torch.empty((0, n * n), dtype=torch.bool)

    moves = torch.tensor(moves, dtype=torch.int64)
    z = torch.tensor(outcomes, dtype=torch.float32)
    game_ids = torch.tensor(game_ids, dtype=torch.int64)

    return x, legal, moves, z, game_ids

# Step 4 - PolicyNet
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    AlphaGo Supervised Learning Policy Network.

    Architecture:
        - Initial 5x5 convolution.
        - (layers - 2) intermediate 3x3 convolutions.
        - Final 1x1 convolution without bias.
        - Learnable position-specific bias.
        - Log-softmax over legal board positions.
    """

    def __init__(self, planes=46, k=32, layers=5, n=5):
        super().__init__()

        self.n = n

        # Store the convolutional layers.
        self.convs = nn.ModuleList()

        # First convolution: input feature planes -> k channels.
        self.convs.append(
            nn.Conv2d(
                in_channels=planes,
                out_channels=k,
                kernel_size=5,
                padding=2
            )
        )

        # Intermediate 3x3 convolutional layers.
        # Each layer preserves the spatial dimensions of the board.
        for _ in range(layers - 2):
            self.convs.append(
                nn.Conv2d(
                    in_channels=k,
                    out_channels=k,
                    kernel_size=3,
                    padding=1
                )
            )

        # Final 1x1 convolution produces one logit per board position.
        # Bias is disabled because a separate position-specific bias
        # is added after flattening.
        self.head = nn.Conv2d(
            in_channels=k,
            out_channels=1,
            kernel_size=1,
            bias=False
        )

        # Independent learnable bias for each board intersection.
        self.position_bias = nn.Parameter(
            torch.zeros(n * n)
        )

    def forward(self, x, legal):
        """
        Compute log-probabilities for all board positions.

        Args:
            x: Feature tensor of shape (B, planes, n, n).
            legal: Boolean legal-move mask of shape (B, n*n).

        Returns:
            Log-probabilities of shape (B, n*n).
            Illegal positions receive -inf.
        """

        # Apply convolutional layers with ReLU activations.
        for conv in self.convs:
            x = F.relu(conv(x))

        # Produce one score for every board intersection.
        x = self.head(x)

        # Flatten spatial dimensions into (B, n*n).
        logits = x.flatten(start_dim=1)

        # Add the learnable position-specific biases.
        logits = logits + self.position_bias

        # Exclude illegal moves from the policy distribution.
        logits = logits.masked_fill(~legal, float("-inf"))

        # Normalize over legal moves using log-softmax.
        return torch.log_softmax(logits, dim=1)

# Step 5 - sl_step
def sl_step(net, optimizer, x, legal, moves):
    """
    Perform one supervised learning step for the policy network.

    Args:
        net: Trained or untrained PolicyNet model.
        optimizer: PyTorch optimizer.
        x: Feature batch of shape (B, 46, n, n).
        legal: Boolean legal-move mask of shape (B, n*n).
        moves: Expert moves of shape (B,), with dtype torch.int64.

    Returns:
        Python float containing the loss value before the optimizer update.
    """

    # Clear gradients from the previous optimization step.
    optimizer.zero_grad()

    # Compute log-probabilities over legal moves.
    log_probs = net(x, legal)

    # Extract the log-probability assigned to each expert move.
    expert_log_probs = log_probs.gather(
        dim=1,
        index=moves.view(-1, 1)
    )

    # Negative mean log-likelihood of the expert moves.
    loss = -expert_log_probs.mean()

    # Preserve the loss value before updating the network.
    loss_value = loss.item()

    # Backpropagate gradients through the policy network.
    loss.backward()

    # Update the network parameters.
    optimizer.step()

    return float(loss_value)

# Step 6 - RolloutPolicy
class RolloutPolicy:
    """
    AlphaGo fast rollout policy.

    Linear softmax model using:
        1. Save-atari features.
        2. Neighbour features relative to the last move.
        3. Local 3x3 pattern features.
    """

    def __init__(self):
        self.index = {}
        self.w = []

    @staticmethod
    def _count(items):
        """Count elements or convert a scalar count to an integer."""
        if items is None:
            return 0

        try:
            return len(items)
        except TypeError:
            return int(items)

    def features(self, game, p):
        """
        Extract active binary features for a candidate move.

        Pattern encoding:
            0 = Outside the board
            1 = Empty point
            1 + L = Friendly stone
            4 + L = Opponent stone

        L is capped at 3.
        """

        n = game.n
        player = game.to_play
        opponent = 3 - player

        row, col = divmod(p, n)

        feature_keys = []

        # --------------------------------------------------
        # 1. Save-atari feature
        # --------------------------------------------------

        orthogonal = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        )

        legal, captured, size, liberties_after = game.move_info(p)

        if legal and self._count(liberties_after) >= 2:

            for dr, dc in orthogonal:
                nr = row + dr
                nc = col + dc

                if not (0 <= nr < n and 0 <= nc < n):
                    continue

                q = nr * n + nc

                if game.board[q] != player:
                    continue

                stones, liberties = game.chain(q)

                if self._count(liberties) == 1:
                    feature_keys.append(("save_atari",))
                    break

        # --------------------------------------------------
        # 2. Neighbour-of-last-move feature
        # --------------------------------------------------

        last_move = game.last_move
        pass_move = game.pass_move

        if last_move is not None and last_move != pass_move:

            last_row, last_col = divmod(last_move, n)

            dr = row - last_row
            dc = col - last_col

            if max(abs(dr), abs(dc)) == 1:
                feature_keys.append(("neighbour", dr, dc))

        # --------------------------------------------------
        # 3. Local 3x3 pattern feature
        # --------------------------------------------------

        offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1)
        )

        pattern = []

        for dr, dc in offsets:

            nr = row + dr
            nc = col + dc

            # Outside the board.
            if not (0 <= nr < n and 0 <= nc < n):
                pattern.append(0)
                continue

            q = nr * n + nc
            stone = game.board[q]

            if stone == 0:

                # Empty intersection.
                pattern.append(1)

            elif stone == player:

                # Friendly stone: 1 + capped liberty count.
                stones, liberties = game.chain(q)

                L = min(self._count(liberties), 3)

                pattern.append(1 + L)

            elif stone == opponent:

                # Opponent stone: 4 + capped liberty count.
                stones, liberties = game.chain(q)

                L = min(self._count(liberties), 3)

                pattern.append(4 + L)

        # The pattern feature is always included.
        feature_keys.append(("pattern", *pattern))

        return feature_keys

    def probs(self, game, moves):
        """
        Calculate softmax probabilities over candidate moves.
        Unknown feature keys contribute zero to the score.
        """

        if not moves:
            return []

        scores = []

        for p in moves:

            score = sum(
                self.w[self.index[key]]
                for key in self.features(game, p)
                if key in self.index
            )

            scores.append(score)

        # Numerically stable softmax.
        max_score = max(scores)

        exp_scores = [
            math.exp(score - max_score)
            for score in scores
        ]

        total = sum(exp_scores)

        return [
            score / total
            for score in exp_scores
        ]

    def train_step(self, game, move, moves, lr):
        """
        Perform one gradient-ascent step on log P(move).

        Previously unseen feature keys are initialized to zero.
        """

        moves = list(moves)

        # Register all features before computing probabilities.
        for p in moves:

            for key in self.features(game, p):

                if key not in self.index:
                    self.index[key] = len(self.w)
                    self.w.append(0.0)

        # Calculate current candidate probabilities.
        probabilities = self.probs(game, moves)

        # Gradient ascent on the expert move's log-probability.
        for p, probability in zip(moves, probabilities):

            gradient = lr * (
                (1.0 if p == move else 0.0) - probability
            )

            for key in self.features(game, p):

                idx = self.index[key]
                self.w[idx] += gradient

    def act(self, game, rng):
        """
        Sample a move from the rollout policy.

        Uses exactly one rng.random() call.
        Passes if no sensible moves remain.
        """

        moves = game.sensible_moves()

        if not moves:
            return game.pass_move

        probabilities = self.probs(game, moves)

        u = rng.random()
        cumulative = 0.0

        for p, probability in zip(moves, probabilities):

            cumulative += probability

            if cumulative > u:
                return p

        # Floating-point rounding fallback.
        return moves[-1]

# Step 7 - sample_moves
def sample_moves(net, games, generator):
    """
    Sample one sensible move for each game using a single batched
    policy-network forward pass.

    Args:
        net: PolicyNet model.
        games: List of unfinished GoGame instances.
        generator: torch.Generator used for reproducible sampling.

    Returns:
        moves: Python list containing one integer move per game.
        x: Float32 feature tensor of shape (B, 46, n, n).
        mask: Boolean tensor of shape (B, n*n).
    """

    # Handle an empty batch.
    if not games:
        return (
            [],
            torch.empty((0, 46, 0, 0), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.bool)
        )

    # ---------------------------------------------------------
    # 1. Extract feature planes and sensible-move masks.
    # ---------------------------------------------------------

    feature_batches = []
    masks = []

    for game in games:

        # Extract the 46 input feature planes.
        planes = feature_planes(game, value=False)

        feature_batches.append(planes)

        # Plane 44 indicates sensible moves:
        # legal empty points that do not fill the player's own eye.
        sensible_mask = planes[44].reshape(-1).bool()

        masks.append(sensible_mask)

    # Stack all games into batched tensors.
    x = torch.stack(feature_batches, dim=0)
    mask = torch.stack(masks, dim=0)

    # ---------------------------------------------------------
    # 2. Initialize every move to pass.
    # Games without sensible moves retain their pass move.
    # ---------------------------------------------------------

    moves = [game.pass_move for game in games]

    # Find games that have at least one sensible move.
    active_indices = torch.nonzero(
        mask.any(dim=1),
        as_tuple=False
    ).flatten()

    # ---------------------------------------------------------
    # 3. Perform one batched policy-network forward pass.
    # Only games with available sensible moves are evaluated.
    # ---------------------------------------------------------

    if active_indices.numel() > 0:

        with torch.no_grad():

            # Compute log-probabilities for all active games
            # in one forward pass.
            log_probs = net(
                x[active_indices],
                mask[active_indices]
            )

            # Convert log-probabilities into sampling weights.
            probabilities = log_probs.exp()

            # -------------------------------------------------
            # 4. Sample one move independently for each game.
            # -------------------------------------------------

            sampled_moves = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator
            ).squeeze(1)

        # Store sampled moves as Python integers.
        for idx, sampled_move in zip(
            active_indices.tolist(),
            sampled_moves.tolist()
        ):
            moves[idx] = int(sampled_move)

    return moves, x, mask

# Step 8 - self_play
def self_play(current, opponent, n_games, generator, n=5, komi=0.5):
    """
    Run minibatch self-play games between the current policy
    network and an earlier opponent policy.

    The current policy plays Black in even-indexed games
    and White in odd-indexed games.

    Only non-pass moves made by the current policy are recorded.

    Args:
        current: Current PolicyNet.
        opponent: Earlier PolicyNet.
        n_games: Number of self-play games.
        generator: torch.Generator for reproducible sampling.
        n: Go board size.
        komi: Komi used by the Go engine.

    Returns:
        x: Float32 tensor of shape (M, 46, n, n).
        mask: Boolean tensor of shape (M, n*n).
        actions: Int64 tensor of shape (M,).
        z: Float32 tensor of shape (M,).
        wins: Number of games won by the current policy.
    """

    # Handle an empty batch.
    if n_games <= 0:
        return (
            torch.empty((0, 46, n, n), dtype=torch.float32),
            torch.empty((0, n * n), dtype=torch.bool),
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.float32),
            0
        )

    # ---------------------------------------------------------
    # 1. Initialize fresh Go games.
    # ---------------------------------------------------------

    games = [
        GoGame(n, komi)
        for _ in range(n_games)
    ]

    # Current policy plays Black in even games and White in odd
    # games. GoGame represents Black as 1 and White as 2.
    current_colors = [
        1 if i % 2 == 0 else 2
        for i in range(n_games)
    ]

    # Store examples separately for each game to preserve
    # game order and move order in the final output.
    game_examples = [
        [] for _ in range(n_games)
    ]

    # ---------------------------------------------------------
    # 2. Play until every game has finished.
    # ---------------------------------------------------------

    while True:

        # Use the Go engine's actual terminal-state property.
        unfinished = [
            i for i, game in enumerate(games)
            if not game.done
        ]

        if not unfinished:
            break

        # -----------------------------------------------------
        # 3. Find unfinished games where current plays.
        # -----------------------------------------------------

        current_indices = [
            i for i in unfinished
            if games[i].to_play == current_colors[i]
        ]

        if current_indices:

            current_games = [
                games[i] for i in current_indices
            ]

            # One batched sampling call for the current policy.
            current_moves, x_batch, mask_batch = sample_moves(
                current,
                current_games,
                generator
            )

            for batch_idx, game_idx in enumerate(current_indices):

                game = games[game_idx]
                move = current_moves[batch_idx]

                # Record the position, legal mask and move only
                # when the current policy makes a non-pass move.
                if move != game.pass_move:

                    game_examples[game_idx].append(
                        (
                            x_batch[batch_idx].clone(),
                            mask_batch[batch_idx].clone(),
                            move
                        )
                    )

                # Play the selected move.
                game.play(move)

        # -----------------------------------------------------
        # 4. Recalculate unfinished games where opponent plays.
        #
        # This is deliberately done AFTER current's moves,
        # because some games may have finished during that batch.
        # -----------------------------------------------------

        unfinished = [
            i for i, game in enumerate(games)
            if not game.done
        ]

        opponent_indices = [
            i for i in unfinished
            if games[i].to_play != current_colors[i]
        ]

        if opponent_indices:

            opponent_games = [
                games[i] for i in opponent_indices
            ]

            # One batched sampling call for the opponent policy.
            opponent_moves, _, _ = sample_moves(
                opponent,
                opponent_games,
                generator
            )

            for game_idx, move in zip(
                opponent_indices,
                opponent_moves
            ):

                game = games[game_idx]

                # Opponent moves are played but not recorded.
                game.play(move)

    # ---------------------------------------------------------
    # 5. Assign final game outcomes.
    # ---------------------------------------------------------

    wins = 0

    xs = []
    masks = []
    actions = []
    outcomes = []

    for game_idx, game in enumerate(games):

        # Determine the winner of the completed game.
        winner = game.winner()

        # Compare the winner with the current policy's colour.
        current_won = (winner == current_colors[game_idx])

        if current_won:
            wins += 1

        # Outcome is always from the current policy's perspective.
        z_game = 1.0 if current_won else -1.0

        # Append this game's examples in their original move order.
        for position, legal_mask, move in game_examples[game_idx]:

            xs.append(position)
            masks.append(legal_mask)
            actions.append(move)
            outcomes.append(z_game)

    # ---------------------------------------------------------
    # 6. Convert recorded examples into correctly typed tensors.
    # ---------------------------------------------------------

    if xs:

        x = torch.stack(xs).to(dtype=torch.float32)

        mask = torch.stack(masks).to(dtype=torch.bool)

    else:

        x = torch.empty(
            (0, 46, n, n),
            dtype=torch.float32
        )

        mask = torch.empty(
            (0, n * n),
            dtype=torch.bool
        )

    actions = torch.tensor(
        actions,
        dtype=torch.int64
    )

    z = torch.tensor(
        outcomes,
        dtype=torch.float32
    )

    return x, mask, actions, z, int(wins)

# Step 9 - train_rl_policy
import copy


def train_rl_policy(sl_net, iterations, n_games, pool_every, lr, generator):
    """
    Train a policy network using REINFORCE self-play against
    a pool of earlier policy versions.

    Args:
        sl_net: Supervised-learning PolicyNet.
        iterations: Number of reinforcement-learning iterations.
        n_games: Number of self-play games per iteration.
        pool_every: Add the current policy to the opponent pool
                    after every pool_every iterations.
        lr: SGD learning rate.
        generator: torch.Generator for reproducible sampling.

    Returns:
        net: Reinforcement-learning policy network.
        pool: List of historical policy state dictionaries.
        history: List of current policy win rates.
    """

    # ---------------------------------------------------------
    # 1. Initialize the reinforcement-learning network.
    # ---------------------------------------------------------

    # Start from an independent copy of the supervised network.
    # The original SL network must remain unchanged.
    net = copy.deepcopy(sl_net)

    # Optimizer for policy-gradient updates.
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr
    )

    # ---------------------------------------------------------
    # 2. Initialize the opponent pool.
    # ---------------------------------------------------------

    # The initial opponent is a copy of the SL network parameters.
    pool = [
        copy.deepcopy(sl_net.state_dict())
    ]

    # Create a separate opponent network.
    opponent = copy.deepcopy(sl_net)

    # Store the current policy's win rate at each iteration.
    history = []

    # ---------------------------------------------------------
    # 3. Reinforcement-learning iterations.
    # ---------------------------------------------------------

    for it in range(1, iterations + 1):

        # Randomly select an opponent from the historical pool.
        k = int(
            torch.randint(
                len(pool),
                (1,),
                generator=generator
            ).item()
        )

        # Load the selected historical parameters into the
        # separate opponent network.
        opponent.load_state_dict(pool[k])

        # -----------------------------------------------------
        # 4. Generate self-play training examples.
        # -----------------------------------------------------

        x, mask, actions, z, wins = self_play(
            net,
            opponent,
            n_games,
            generator
        )

        # -----------------------------------------------------
        # 5. REINFORCE policy-gradient update.
        # -----------------------------------------------------

        optimizer.zero_grad()

        # Calculate log-probabilities of the recorded moves.
        log_probs = net(x, mask)

        # Extract the log-probability assigned to each
        # action actually selected during self-play.
        selected_log_probs = log_probs.gather(
            dim=1,
            index=actions.unsqueeze(1)
        ).squeeze(1)

        # REINFORCE objective:
        # Minimize the negative outcome-weighted log-likelihood.
        #
        # Positive outcomes encourage the selected actions.
        # Negative outcomes discourage the selected actions.
        loss = -(
            z * selected_log_probs
        ).sum() / n_games

        # Backpropagate the policy-gradient loss.
        loss.backward()

        # Update the reinforcement-learning policy.
        optimizer.step()

        # -----------------------------------------------------
        # 6. Record the current policy's win rate.
        # -----------------------------------------------------

        history.append(wins / n_games)

        # -----------------------------------------------------
        # 7. Update the historical opponent pool.
        # -----------------------------------------------------

        if it % pool_every == 0:

            # Save an independent snapshot of the current policy.
            pool.append(
                copy.deepcopy(net.state_dict())
            )

    return net, pool, history

