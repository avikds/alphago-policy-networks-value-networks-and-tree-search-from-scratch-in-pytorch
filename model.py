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

