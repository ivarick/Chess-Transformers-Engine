import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import math
import chess
import chess.pgn
from tqdm import tqdm  
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt  
import time


os.environ['OMP_NUM_THREADS'] = '6'  
os.environ['MKL_NUM_THREADS'] = '6' 
os.environ['NUMEXPR_NUM_THREADS'] = '6'  
torch.set_num_threads(6)  
torch.backends.mkl.enabled = True if torch.backends.mkl.is_available() else False


from model import TransformerChessModel

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

RESULT_TO_LABEL = {
    "1-0": 0,  # White win
    "0-1": 1,  # Black win
    "1/2-1/2": 2  # Draw
}


def board_to_features_enhanced(board):
    """enhanced feature encoding for chess positions with additional game-specific features"""

    features = np.zeros((22, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        piece_idx = PIECE_TO_INDEX[piece.symbol()]
        features[piece_idx, rank, file] = 1
    

    features[12, :, :] = 1.0 if board.turn else 0.0
    
    # material value map (index 13)
    piece_values = {'P': 1.0, 'N': 3.0, 'B': 3.0, 'R': 5.0, 'Q': 9.0, 'K': 0.0,
                    'p': -1.0, 'n': -3.0, 'b': -3.0, 'r': -5.0, 'q': -9.0, 'k': 0.0}
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        features[13, rank, file] = piece_values[piece.symbol()] / 9.0  
    
    # attack maps (indices 14-15)
    white_attacks = np.zeros((8, 8), dtype=np.float32)
    black_attacks = np.zeros((8, 8), dtype=np.float32)
    
    for square in chess.SQUARES:
        # count white attacks
        if board.is_attacked_by(chess.WHITE, square):
            rank, file = divmod(square, 8)
            white_attacks[rank, file] += 1
        
        # count black attacks
        if board.is_attacked_by(chess.BLACK, square):
            rank, file = divmod(square, 8)
            black_attacks[rank, file] += 1
    

    max_attacks = max(np.max(white_attacks), np.max(black_attacks)) if max(np.max(white_attacks), np.max(black_attacks)) > 0 else 1
    features[14, :, :] = white_attacks / max_attacks
    features[15, :, :] = black_attacks / max_attacks

    features[16, :, :] = 1.0 if board.is_check() else 0.0
    

    legal_move_count = len(list(board.legal_moves))
    board.push(chess.Move.null())  
    opponent_move_count = len(list(board.legal_moves))
    board.pop() 
    

    features[17, :, :] = legal_move_count / 40.0
    features[18, :, :] = opponent_move_count / 40.0
    
    # pawn structure (indices 19-20)
    # white and black pawn maps
    white_pawns = np.zeros((8, 8), dtype=np.float32)
    black_pawns = np.zeros((8, 8), dtype=np.float32)
    
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        if piece.symbol() == 'P':
            white_pawns[rank, file] = 1.0
        elif piece.symbol() == 'p':
            black_pawns[rank, file] = 1.0
    
    features[19, :, :] = white_pawns
    features[20, :, :] = black_pawns
    
    # king safety (index 21) - based on attacks near king
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    if white_king_square is not None:
        white_king_rank, white_king_file = divmod(white_king_square, 8)
        white_king_safety = 0
        for r in range(max(0, white_king_rank-1), min(8, white_king_rank+2)):
            for f in range(max(0, white_king_file-1), min(8, white_king_file+2)):
                sq = r * 8 + f
                if board.is_attacked_by(chess.BLACK, sq):
                    white_king_safety -= 1
        
       
        features[21, :, :] = max(-3, white_king_safety) / -3.0
    
    if black_king_square is not None:
        black_king_rank, black_king_file = divmod(black_king_square, 8)
        black_king_safety = 0
        for r in range(max(0, black_king_rank-1), min(8, black_king_rank+2)):
            for f in range(max(0, black_king_file-1), min(8, black_king_file+2)):
                sq = r * 8 + f
                if board.is_attacked_by(chess.WHITE, sq):
                    black_king_safety -= 1
        
       
        if not board.turn:
            features[21, :, :] = max(-3, black_king_safety) / -3.0
    
    return torch.tensor(features, dtype=torch.float32)


def predict_best_move(model, board=None, fen_string=None, top_k=5, depth=2, temperature=0.5, verbose=True):

    start = time.time()
    device = next(model.parameters()).device
    
    evaluator = MaterialLossEvaluator(model)
    
    def evaluate_position(board):
        nonlocal nodes_evaluated
        nodes_evaluated += 1
        return evaluator.evaluate(board)

    
    if not board:
        board = chess.Board(fen_string) if fen_string else chess.Board()
    
    if verbose:
        print(f"\nanalyzing {'White' if board.turn else 'Black'} to move. Depth: {depth}")
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []
    

    transposition_table = {}
    nodes_evaluated = 0
    

    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
                    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0}
    
    def evaluate_position(board):
        nonlocal nodes_evaluated
        nodes_evaluated += 1
        features = board_to_features_enhanced(board).unsqueeze(0).to(device)
        return model(features).item()
    
    def get_move_priority(move):
        priority = 0
        
      
        if board.is_capture(move):
            victim_square = move.to_square
            victim_piece = board.piece_at(victim_square)
            attacker_square = move.from_square
            attacker_piece = board.piece_at(attacker_square)
            
            if victim_piece and attacker_piece:
                victim_value = abs(piece_values[victim_piece.symbol()])
                attacker_value = abs(piece_values[attacker_piece.symbol()])
         
                priority += 10 + (victim_value * 10 - attacker_value)
        

        if move.promotion:
            priority += 9
        

        if board.gives_check(move):
            priority += 5
        

        if board.is_castling(move):
            priority += 3
        

        if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.PAWN:
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            

            if 2 <= to_file <= 5 and 3 <= to_rank <= 4:
                priority += 1
        
        return priority
    
    def get_board_hash(board):
       return board.fen()
    
    def minimax_with_pruning(board, depth, alpha, beta, maximizing):

        board_hash = get_board_hash(board)
        key = (board_hash, depth, maximizing)
        
        if key in transposition_table:
            return transposition_table[key]
        

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0 if maximizing else -1.0
            elif result == "0-1":
                return -1.0 if maximizing else 1.0
            else:  
                return 0.0
        
        if depth == 0:
            eval_score = evaluate_position(board)
            transposition_table[key] = eval_score
            return eval_score
        

        moves = list(board.legal_moves)
        moves.sort(key=get_move_priority, reverse=True)
        
        best_value = float('-inf') if maximizing else float('inf')
        for move in moves:
            board.push(move)
            

            value = minimax_with_pruning(board, depth - 1, alpha, beta, not maximizing)
            board.pop()
            

            if maximizing:
                best_value = max(best_value, value)
                alpha = max(alpha, value)
            else:
                best_value = min(best_value, value)
                beta = min(beta, value)
            

            if beta <= alpha:
                break
        

        transposition_table[key] = best_value
        return best_value
    

    move_evaluations = []
    

    ordered_moves = sorted(legal_moves, key=get_move_priority, reverse=True)
    
    for i, move in enumerate(ordered_moves):
        board_copy = board.copy(stack=False)
        board_copy.push(move)
        

        score = minimax_with_pruning(
            board_copy, 
            depth - 1, 
            float('-inf'), 
            float('inf'), 
            not board.turn
        )
        
        move_san = board.san(move)
        move_evaluations.append((move, score, move_san))
        
        if verbose and (i % 5 == 0 or i == len(ordered_moves) - 1):
            elapsed = time.time() - start
            print(f"Move {move_san}: {score:.4f} (Progress: {100*(i+1)/len(legal_moves):.1f}%)")
    

    sorted_moves = sorted(
        move_evaluations,
        key=lambda x: (x[1], get_move_priority(x[0])),
        reverse=board.turn
    )

    

    final_moves = sorted_moves[:top_k]
    
    if verbose:
        print(f"\ncompleted in {time.time() - start:.2f}s — nodes evaluated: {nodes_evaluated}")
        for i, (_, score, san) in enumerate(final_moves):
            print(f"{i+1}. {san}: {score:.4f}")
    
    return [(m, s, san) for m, s, san in final_moves]

def get_material_balance(board):
    
    values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
    white_material = sum(values[p.symbol()] for p in board.piece_map().values() if p.symbol().isupper())
    black_material = sum(values[p.symbol()] for p in board.piece_map().values() if p.symbol().islower())
    return white_material - black_material

class MaterialLossEvaluator:
    def __init__(self, base_model, weight=0.3):
        self.model = base_model
        self.weight = weight 
        self.device = next(model.parameters()).device
    
    def evaluate(self, board):
  
        features = board_to_features_enhanced(board).unsqueeze(0).to(self.device)
        model_eval = self.model(features).item()
        

        material_balance = get_material_balance(board)
        material_eval = sigmoid(material_balance)  
        
   
        combined_eval = (1 - self.weight) * model_eval + self.weight * material_eval
        return combined_eval

def print_gpu_stats():
    if torch.cuda.is_available():
        try:
    
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB")
            try:
                utilization = torch.cuda.utilization()
                print(f"GPU Utilization: {utilization}%")
            except Exception as e:
                print(f"GPU Utilization: Not available (NVML error: {type(e).__name__})")
                
        except Exception as e:
            print(f"GPU stats unavailable: {e}")
    else:
        print("CUDA not available")




def sigmoid(x):
   
    return 1 / (1 + math.exp(-x * 0.5))  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TransformerChessModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval() 




fen = "2r1k2r/1p6/p3b3/q2pNp1p/1b1n1P2/P7/1PBN1PP1/1QKR3R b k - 3 20"
board = chess.Board(fen)

best_moves = predict_best_move(
    model=model,
    board=board,
    top_k=3,      
    depth=2,      
    temperature=1,
    verbose=True  
)

for move, score, san in best_moves:
    print(f"{san} with score {score:.4f}")


total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
   
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

count_parameters(model)

