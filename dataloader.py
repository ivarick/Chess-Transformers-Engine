import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm 
import gc
import queue
import threading

class AsyncDataLoader:
    def __init__(self, dataset, batch_size, num_workers=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue = queue.Queue(maxsize=50)
        
    def start_loading(self):
       
        for _ in range(self.num_workers):
            thread = threading.Thread(target=self._load_batches)
            thread.daemon = True
            thread.start()
    
    def _load_batches(self):
        while True:
          
            batch = self._prepare_batch()
            self.queue.put(batch)
    
    def __iter__(self):
        while True:
            yield self.queue.get()


PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

RESULT_TO_LABEL = {
    "1-0": 0,  # white win
    "0-1": 1,  # black win
    "1/2-1/2": 2  # draw
}

class ChessDataset(Dataset):
    def __init__(self, pgn_files, max_games_per_file=10, use_enhanced=True, preload_in_memory=True):
        self.pgn_files = pgn_files
        self.max_games_per_file = max_games_per_file
        self.use_enhanced = use_enhanced
        self.game_positions = []
        self.index_files()
        

        self.file_handle_cache = {}
        self.max_open_files = 5
    
    def index_files(self):
        for pgn_file in self.pgn_files:
            print(f"indexing games from {pgn_file}...")
            
            with open(pgn_file, 'r') as f:
                game_idx = 0
                pbar = tqdm(total=self.max_games_per_file, desc="games indexed", unit="game")
                
                while game_idx < self.max_games_per_file:
                    game_start_pos = f.tell()  
                    
          
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    

                    result = game.headers.get("result", "*")
                    game_result = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result, 0.5)

                    board = chess.Board()
                    move_count = 0
                    
                    node = game
                    while node.variations and move_count < 100:
                        next_node = node.variations[0]
                        move = next_node.move
                        board.push(move)
                        
                        if board.fullmove_number > 10:
                    
                            self.game_positions.append((pgn_file, game_start_pos, move_count, game_result))
                        
                        move_count += 1
                        node = next_node
                    
                    game_idx += 1
                    pbar.update(1)
                
                pbar.close()
            print(f" indexed {game_idx} games from {pgn_file}")
        
        gc.collect()
    
    def __len__(self):
        return len(self.game_positions)
    
    def _get_file_handle(self, file_path):
        if file_path in self.file_handle_cache:
            return self.file_handle_cache[file_path]
        
        if len(self.file_handle_cache) >= self.max_open_files:
            oldest_file = next(iter(self.file_handle_cache))
            self.file_handle_cache[oldest_file].close()
            del self.file_handle_cache[oldest_file]
        

        file_handle = open(file_path, 'r')
        self.file_handle_cache[file_path] = file_handle
        return file_handle
    
    def __getitem__(self, idx):
        file_path, game_start_pos, target_move_idx, result = self.game_positions[idx]
        
        f = self._get_file_handle(file_path)
        f.seek(game_start_pos)   
        game = chess.pgn.read_game(f) 
        board = chess.Board()
        current_move = 0
        node = game
        while node.variations and current_move <= target_move_idx:
            next_node = node.variations[0]
            move = next_node.move
            board.push(move)
            current_move += 1
            
            if current_move > target_move_idx:
                break
                
            node = next_node
        
        if self.use_enhanced:
            features = self._board_to_features_enhanced(board)
        else:
            features = self._board_to_features(board)
            
        return features, torch.tensor(result, dtype=torch.float32)
    
    def _board_to_features(self, board):
        features = np.zeros((22, 8, 8), dtype=np.float32)
        
        
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            piece_idx = "PNBRQKpnbrqk".index(piece.symbol())
            features[piece_idx, rank, file] = 1
        

        features[12, :, :] = float(board.turn)
     
        features[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        features[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        features[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        features[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
        
        # En passant
        if board.ep_square is not None:
            ep_rank, ep_file = divmod(board.ep_square, 8)
            features[17, ep_rank, ep_file] = 1
               
        features[18, :, :] = min(1.0, board.halfmove_clock / 100.0)     
        features[19, :, :] = min(1.0, board.fullmove_number / 100.0)
        features[20, :, :] = float(board.is_check())
        features[21, :, :] = min(1.0, len(list(board.legal_moves)) / 40.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _board_to_features_enhanced(self, board):
        """enhanced feature extraction with 22 channels to match model expectations"""
        return self._board_to_features(board)  
    
    def __del__(self):
        for file_handle in self.file_handle_cache.values():
            file_handle.close()


class PatchEmbedding(nn.Module):
    """convert the chess board into patches and embed them."""
    def __init__(self, in_channels=22, embed_dim=512, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 16, embed_dim]
        x = self.norm(x)
        return x

