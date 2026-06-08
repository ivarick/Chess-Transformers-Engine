"""Dataset utilities for PGN-based chess position training."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from features import board_to_features

RESULT_TO_SCORE = {
    "1-0": 1.0,
    "1/2-1/2": 0.5,
    "0-1": 0.0,
}


@dataclass(frozen=True)
class IndexedPosition:
    pgn_path: Path
    game_offset: int
    ply_index: int
    result_score: float


class ChessDataset(Dataset):
    """Index PGN games and replay positions on demand.

    The dataset stores file offsets instead of all boards, which keeps memory
    usage predictable even for large PGN files.
    """

    def __init__(
        self,
        pgn_files: list[str | Path],
        max_games_per_file: int | None = 2_000_000,
        max_plies_per_game: int = 100,
        min_fullmove_number: int = 10,
        show_progress: bool = True,
        encoding: str = "utf-8",
        use_enhanced: bool | None = None,
        preload_in_memory: bool | None = None,
    ):
        # Compatibility arguments from the original dataset API. Feature
        # extraction is now centralized in features.py.
        _ = (use_enhanced, preload_in_memory)

        self.pgn_files = [Path(path) for path in pgn_files]
        self.max_games_per_file = max_games_per_file
        self.max_plies_per_game = max_plies_per_game
        self.min_fullmove_number = min_fullmove_number
        self.show_progress = show_progress
        self.encoding = encoding
        self.positions: list[IndexedPosition] = []
        self.file_handle_cache: OrderedDict[Path, TextIO] = OrderedDict()
        self.max_open_files = 5

        self.index_files()

    def index_files(self) -> None:
        for pgn_path in self.pgn_files:
            if not pgn_path.exists():
                raise FileNotFoundError(f"PGN file not found: {pgn_path}")

            print(f"Indexing games from {pgn_path}...")
            with pgn_path.open("r", encoding=self.encoding, errors="replace") as handle:
                game_count = 0
                progress = tqdm(
                    total=self.max_games_per_file,
                    desc=f"Indexing {pgn_path.name}",
                    unit="game",
                    disable=not self.show_progress,
                )

                try:
                    while self.max_games_per_file is None or game_count < self.max_games_per_file:
                        game_offset = handle.tell()
                        game = chess.pgn.read_game(handle)
                        if game is None:
                            break

                        self._index_game(pgn_path, game, game_offset)
                        game_count += 1
                        progress.update(1)
                finally:
                    progress.close()

            print(f"Indexed {game_count} games and {len(self.positions)} total positions.")

    def _index_game(self, pgn_path: Path, game: chess.pgn.Game, game_offset: int) -> None:
        result_score = RESULT_TO_SCORE.get(game.headers.get("Result", "*"), 0.5)
        board = chess.Board()
        node = game

        for ply_index in range(self.max_plies_per_game):
            if not node.variations:
                break

            node = node.variations[0]
            board.push(node.move)

            if board.fullmove_number > self.min_fullmove_number:
                self.positions.append(IndexedPosition(pgn_path, game_offset, ply_index, result_score))

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        position = self.positions[index]
        board = self._replay_position(position)
        target = torch.tensor(position.result_score, dtype=torch.float32)
        return board_to_features(board), target

    def _replay_position(self, position: IndexedPosition) -> chess.Board:
        handle = self._get_file_handle(position.pgn_path)
        handle.seek(position.game_offset)

        game = chess.pgn.read_game(handle)
        if game is None:
            raise RuntimeError(f"Could not replay PGN game at offset {position.game_offset}")

        board = chess.Board()
        node = game
        for _ in range(position.ply_index + 1):
            if not node.variations:
                break
            node = node.variations[0]
            board.push(node.move)

        return board

    def _get_file_handle(self, pgn_path: Path) -> TextIO:
        if pgn_path in self.file_handle_cache:
            self.file_handle_cache.move_to_end(pgn_path)
            return self.file_handle_cache[pgn_path]

        if len(self.file_handle_cache) >= self.max_open_files:
            _, oldest_handle = self.file_handle_cache.popitem(last=False)
            oldest_handle.close()

        handle = pgn_path.open("r", encoding=self.encoding, errors="replace")
        self.file_handle_cache[pgn_path] = handle
        return handle

    def close(self) -> None:
        for file_handle in self.file_handle_cache.values():
            file_handle.close()
        self.file_handle_cache.clear()

    def __del__(self) -> None:
        self.close()
