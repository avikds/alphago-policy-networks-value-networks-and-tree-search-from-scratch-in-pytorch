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

