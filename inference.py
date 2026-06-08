"""Run move search with a trained transformer chess evaluator."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn

from features import board_to_features
from model import TransformerChessModel, count_parameters

EXAMPLE_FEN = "2r1k2r/1p6/p3b3/q2pNp1p/1b1n1P2/P7/1PBN1PP1/1QKR3R b k - 3 20"

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass(frozen=True)
class MoveEvaluation:
    move: chess.Move
    san: str
    score: float


class MaterialAwareEvaluator:
    """Blend neural evaluation with a simple material prior."""

    def __init__(self, model: nn.Module, material_weight: float = 0.25):
        if not 0.0 <= material_weight <= 1.0:
            raise ValueError("material_weight must be between 0 and 1")

        self.model = model
        self.material_weight = material_weight
        self.device = next(model.parameters()).device

    def evaluate(self, board: chess.Board) -> float:
        features = board_to_features(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            model_score = float(self.model(features).squeeze().item())

        material_score = sigmoid(material_balance(board) * 0.5)
        return (1.0 - self.material_weight) * model_score + self.material_weight * material_score


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def material_balance(board: chess.Board) -> float:
    """Return material balance from White's perspective."""

    balance = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        balance += value if piece.color == chess.WHITE else -value
    return balance


def terminal_score(board: chess.Board) -> float | None:
    """Return a final score from White's perspective, or None if not terminal."""

    if board.is_checkmate():
        return 0.0 if board.turn == chess.WHITE else 1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
        return 0.5
    return None


def move_priority(board: chess.Board, move: chess.Move) -> float:
    priority = 0.0

    if board.is_capture(move):
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, not board.turn)
        if attacker and victim:
            priority += 10.0 + PIECE_VALUES[victim.piece_type] * 10.0 - PIECE_VALUES[attacker.piece_type]

    if move.promotion:
        priority += PIECE_VALUES.get(move.promotion, 0)
    if board.gives_check(move):
        priority += 5.0
    if board.is_castling(move):
        priority += 3.0

    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    if 2 <= to_file <= 5 and 3 <= to_rank <= 4:
        priority += 1.0

    return priority


def minimax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    evaluator: MaterialAwareEvaluator,
    table: dict[tuple[str, int], float],
) -> float:
    final = terminal_score(board)
    if final is not None:
        return final
    if depth <= 0:
        return evaluator.evaluate(board)

    key = (board.fen(), depth)
    if key in table:
        return table[key]

    maximizing = board.turn == chess.WHITE
    moves = sorted(board.legal_moves, key=lambda move: move_priority(board, move), reverse=True)

    if maximizing:
        best_value = float("-inf")
        for move in moves:
            board.push(move)
            best_value = max(best_value, minimax(board, depth - 1, alpha, beta, evaluator, table))
            board.pop()
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
    else:
        best_value = float("inf")
        for move in moves:
            board.push(move)
            best_value = min(best_value, minimax(board, depth - 1, alpha, beta, evaluator, table))
            board.pop()
            beta = min(beta, best_value)
            if beta <= alpha:
                break

    table[key] = best_value
    return best_value


def predict_best_moves(
    model: nn.Module,
    board: chess.Board,
    top_k: int = 5,
    depth: int = 2,
    material_weight: float = 0.25,
    verbose: bool = True,
) -> list[MoveEvaluation]:
    start = time.time()
    model.eval()
    evaluator = MaterialAwareEvaluator(model, material_weight=material_weight)
    table: dict[tuple[str, int], float] = {}

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    evaluations: list[MoveEvaluation] = []
    ordered_moves = sorted(legal_moves, key=lambda move: move_priority(board, move), reverse=True)

    for index, move in enumerate(ordered_moves, start=1):
        board.push(move)
        score = minimax(board, depth - 1, float("-inf"), float("inf"), evaluator, table)
        board.pop()
        evaluations.append(MoveEvaluation(move=move, san=board.san(move), score=score))

        if verbose and (index % 5 == 0 or index == len(ordered_moves)):
            progress = 100.0 * index / len(ordered_moves)
            print(f"Analyzed {index:>3}/{len(ordered_moves)} moves ({progress:5.1f}%).")

    evaluations.sort(key=lambda item: (item.score, move_priority(board, item.move)), reverse=board.turn)

    if verbose:
        elapsed = time.time() - start
        print(f"\nCompleted depth-{depth} search in {elapsed:.2f}s.")
        for rank, evaluation in enumerate(evaluations[:top_k], start=1):
            print(f"{rank}. {evaluation.san:<8} score={evaluation.score:.4f}")

    return evaluations[:top_k]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_model(checkpoint_path: Path | None, device: torch.device) -> TransformerChessModel:
    checkpoint = None
    config = {}
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            config = checkpoint.get("config", {})

    model = TransformerChessModel(
        embed_dim=int(config.get("embed_dim", 512)),
        num_blocks=int(config.get("num_blocks", 4)),
        num_heads=int(config.get("num_heads", 8)),
        ff_dim=int(config.get("ff_dim", 2048)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)

    if checkpoint is None:
        print("No checkpoint supplied; using randomly initialized weights.")
        return model

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fen", default=EXAMPLE_FEN, help="FEN string to analyze.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a .pth checkpoint.")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--material-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    board = chess.Board(args.fen)
    model = load_model(args.checkpoint, device)

    total_params, trainable_params = count_parameters(model)
    print(f"Using device: {device}")
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")
    print(f"Position: {board.fen()}")
    print(f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}")

    moves = predict_best_moves(
        model,
        board,
        top_k=args.top_k,
        depth=args.depth,
        material_weight=args.material_weight,
        verbose=not args.quiet,
    )

    if args.quiet:
        for evaluation in moves:
            print(f"{evaluation.san}\t{evaluation.score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
