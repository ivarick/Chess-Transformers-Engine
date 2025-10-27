import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import chess
import chess.pgn
import traceback
import matplotlib.pyplot as plt  
import gc
import psutil
from tqdm import tqdm  
from torch.utils.data import DataLoader, random_split

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ['OMP_NUM_THREADS'] = '6'  
os.environ['MKL_NUM_THREADS'] = '6' 
os.environ['NUMEXPR_NUM_THREADS'] = '6'  

torch.set_num_threads(6)  
torch.backends.mkl.enabled = True if torch.backends.mkl.is_available() else False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()  


from model import TransformerChessModel
from dataloader import ChessDataset



def optimize_memory():
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)  
    
    gc.set_threshold(700, 10, 10) 
    torch.backends.cudnn.benchmark = True

p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

RESULT_TO_LABEL = {
    "1-0": 0,  # white win
    "0-1": 1,  # black win
    "1/2-1/2": 2  # draw
}

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
# THE DATASET IS A PGN FILES
class ChessDataset(Dataset): #index 2 million games from the chosen dataset 
    def __init__(self, pgn_files, max_games_per_file=2000000, use_enhanced=True, preload_in_memory=True): 
        # for faster indexing and efficient memory use
        self.pgn_files = pgn_files
        self.max_games_per_file = max_games_per_file
        self.use_enhanced = use_enhanced
        self.game_positions = []
        self.index_files()
        self.file_handle_cache = {}
        self.max_open_files = 5
    
    def index_files(self):
        """create an index of positions without loading all boards into memory"""
        for pgn_file in self.pgn_files:
            print(f"indexing games from {pgn_file}...")
            
            with open(pgn_file, 'r') as f:
                game_idx = 0
                pbar = tqdm(total=self.max_games_per_file, desc="games Indexed", unit="game")
                
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
            print(f"indexed {game_idx} games from {pgn_file}")
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
        """basic feature extraction - maintain 22 channels to match model expectations"""
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
        
        # en passant
        if board.ep_square is not None:
            ep_rank, ep_file = divmod(board.ep_square, 8)
            features[17, ep_rank, ep_file] = 1
               
        features[18, :, :] = min(1.0, board.halfmove_clock / 100.0)     
    
        features[19, :, :] = min(1.0, board.fullmove_number / 100.0)
 
        features[20, :, :] = float(board.is_check())
 
        features[21, :, :] = min(1.0, len(list(board.legal_moves)) / 40.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _board_to_features_enhanced(self, board):
        return self._board_to_features(board)  
    
    def __del__(self):
        """clean up file handles on destruction"""
        for file_handle in self.file_handle_cache.values():
            file_handle.close()
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=22, embed_dim=512, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [b, embed_dim, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [b, 16, embed_dim]
        x = self.norm(x)
        return x
    

def train_model(model, pgn_files, batch_size=256, epochs=20, learning_rate=0.00005, weight_decay=0.01, shuffle=True, resume_training=True):
 
    print(model)
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    

    def print_gpu_stats():
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            print(f"GPU Utilization: {torch.cuda.utilization()}%")
    
   

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    checkpoint_train_losses = []
    checkpoint_val_losses = []
    checkpoint_epochs = []
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    
  
    dataset = ChessDataset(pgn_files)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.0,
        div_factor=25,
        final_div_factor=1000
    )
    
    checkpoint_path = "ultron.pth"
    if resume_training and os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
           
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
       
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
               
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                
                if 'train_losses' in checkpoint:
                    train_losses = checkpoint['train_losses']
                if 'val_losses' in checkpoint:
                    val_losses = checkpoint['val_losses']
                if 'checkpoint_train_losses' in checkpoint:
                    checkpoint_train_losses = checkpoint['checkpoint_train_losses']
                if 'checkpoint_val_losses' in checkpoint:
                    checkpoint_val_losses = checkpoint['checkpoint_val_losses']
                if 'checkpoint_epochs' in checkpoint:
                    checkpoint_epochs = checkpoint['checkpoint_epochs']
                
                print(f"resumed training from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
                
             
                if start_epoch > 0:
                 
                    steps_completed = start_epoch * len(train_loader)
                    for _ in range(steps_completed):
                 
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_loss)
                
            else:
          
                model.load_state_dict(checkpoint)
                print("loaded model weights only. Starting fresh training state.")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting with fresh model and training state.")
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        print("no existing checkpoint found or resume_training=False. Starting with fresh weights.")
    
 
    try:
        print("testing batch loading...")
        test_batch = next(iter(train_loader))
        print(f"successfully loaded a batch of shape: {test_batch[0].shape}")
        
        print("testing batch iteration...")
        for idx, (inputs, targets) in enumerate(train_loader):
            if idx < 3:
                print(f"batch {idx} loaded: input shape {inputs.shape}, target shape {targets.shape}")
            else:
                print("first 3 batches loaded successfully!")
                break
                
    except Exception as e:
        print(f"error loading data: {e}")
        traceback.print_exc()
        return model
    
    torch.backends.cudnn.benchmark = True
    criterion = nn.MSELoss()
    print(f"starting training from epoch {start_epoch + 1} to {epochs}")
    for epoch in range(start_epoch, epochs):

        model.train()
        train_loss = 0.019
        samples_processed = 0
        checkpoint_interval = max(1, len(train_loader) // 10) 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
          
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                smoothing = 0.05
                targets = targets * (1 - smoothing) + 0.5 * smoothing
                loss = criterion(outputs.squeeze(), targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
                
            batch_loss = loss.item() * inputs.size(0)
            train_loss += batch_loss
            samples_processed += inputs.size(0)
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
    
            if (batch_idx + 1) % checkpoint_interval == 0 or batch_idx == len(train_loader) - 1:
                current_train_loss = train_loss / samples_processed
                if batch_idx % 100 == 0:
                 print_gpu_stats()
                 for batch_idx, (inputs, targets) in enumerate(train_loader):
                   inputs = inputs.to(device, non_blocking=True) 
                   targets = targets.to(device, non_blocking=True)
    
                model.eval()
                val_loss = 0.0
                val_samples = 0
                
                subset_size = max(1, len(val_loader) // 4)
                with torch.no_grad():
                    for i, (val_inputs, val_targets) in enumerate(val_loader):
                        if i >= subset_size:
                            break
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_outputs = model(val_inputs)
                        val_batch_loss = criterion(val_outputs.squeeze(), val_targets).item() * val_inputs.size(0)
                        val_loss += val_batch_loss
                        val_samples += val_inputs.size(0)
                
                current_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
                model.train()
                
                current_epoch = epoch + (batch_idx + 1) / len(train_loader)
                checkpoint_epochs.append(current_epoch)
                checkpoint_train_losses.append(current_train_loss)
                checkpoint_val_losses.append(current_val_loss)
                print(f"\nCheckpoint | Progress: {100*(batch_idx+1)/len(train_loader):.1f}% | "
                      f"Train Loss: {current_train_loss:.4f} | Val Loss: {current_val_loss:.4f}")
                
             
                checkpoint_data = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'checkpoint_epochs': checkpoint_epochs,
                    'checkpoint_train_losses': checkpoint_train_losses,
                    'checkpoint_val_losses': checkpoint_val_losses,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size
                }
        
                torch.save(checkpoint_data, checkpoint_path)
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save(model.state_dict(), "ultron_best.pth")
                    print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
                
   
                if len(checkpoint_epochs) > 1:
                    plt.figure(figsize=(12, 7))
                    plt.plot(checkpoint_epochs, checkpoint_train_losses, 'b-o', label='Training Loss', alpha=0.7)
                    plt.plot(checkpoint_epochs, checkpoint_val_losses, 'r-o', label='Validation Loss', alpha=0.7)
                    
        
                    for e in range(start_epoch + 1, epoch + 2):
                        plt.axvline(x=e, color='gray', linestyle='--', alpha=0.5)
                    
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss (Checkpoint Level)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('checkpoint_progress_plot.png', dpi=100)
                    plt.close()
        
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        

        checkpoint_data = {
            'epoch': epoch,
            'batch_idx': len(train_loader) - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'checkpoint_epochs': checkpoint_epochs,
            'checkpoint_train_losses': checkpoint_train_losses,
            'checkpoint_val_losses': checkpoint_val_losses,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size
        }
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TransformerChessModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train() 

pgn_files = ["lichess_db_standard_rated_2014-08.pgn"]  

train_model(model, pgn_files, epochs=10, learning_rate=0.00001, batch_size=32, shuffle=True)
