"""Shared chess-board feature extraction for training and inference."""

from __future__ import annotations

import chess
import numpy as np
import torch

NUM_FEATURE_PLANES = 22

PIECE_TO_CHANNEL = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def board_to_features(board: chess.Board) -> torch.Tensor:
    """Encode a python-chess board as a 22x8x8 float tensor.

    The output contract is intentionally shared by the dataset and inference
    code. Changing this layout requires retraining or adapting model weights.
    """

    features = np.zeros((NUM_FEATURE_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        features[PIECE_TO_CHANNEL[piece.symbol()], rank, file] = 1.0

    features[12, :, :] = float(board.turn == chess.WHITE)
    features[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        features[17, ep_rank, ep_file] = 1.0

    features[18, :, :] = min(1.0, board.halfmove_clock / 100.0)
    features[19, :, :] = min(1.0, board.fullmove_number / 100.0)
    features[20, :, :] = float(board.is_check())
    features[21, :, :] = min(1.0, board.legal_moves.count() / 40.0)

    return torch.from_numpy(features)
