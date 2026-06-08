import chess
import torch
import torch.nn as nn

from inference import material_balance, predict_best_moves


class ConstantModel(nn.Module):
    def __init__(self, value: float = 0.5):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand(x.shape[0], 1)


def test_material_balance_is_white_perspective():
    board = chess.Board("8/8/8/8/8/8/8/Q3k2K w - - 0 1")

    assert material_balance(board) == 9


def test_predict_best_moves_returns_legal_san_moves():
    board = chess.Board()
    model = ConstantModel()

    moves = predict_best_moves(model, board, top_k=3, depth=1, verbose=False)

    assert len(moves) == 3
    assert all(move.move in board.legal_moves for move in moves)
    assert all(move.san for move in moves)
