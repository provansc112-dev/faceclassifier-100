import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import json
import config
import model as model_loader
import utils

class Trainer:
    def __init__(self, model_type):
        self.model_type = model_type
        self.device = config.DEVICE
        
        # 1. Load Data
        print(f"Loading data for {model_type.upper()}...")
        self.dataloaders, self.dataset_sizes = utils.get_dataloaders(model_type)
        if not self.dataloaders:
            raise RuntimeError("Failed to load data.")

        # 2. Load Model
        self.model = model_loader.load_trained_model(model_type, load_best=False)
        
        # 3. Loss Function
        self.criterion = nn.CrossEntropyLoss()
        
    def _run_epoch(self, optimizer, phase):
        self.model.train() if phase == 'train' else self.model.eval()
        running_loss, running_corrects = 0.0, 0
        
        pbar = tqdm(self.dataloaders[phase], desc=f"[{phase.upper()}]", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(images)
                
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[-1]
                else:
                    logits = outputs
                
                _, preds = torch.max(logits, 1)
                loss = self.criterion(logits, labels.long())

                if phase == 'train':
                    loss.backward()
                    if self.model_type == 'swin':
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
        
        return epoch_loss, epoch_acc

    def train_swin(self):
        print(f"\nSTARTING SWIN TRAINING ({config.SWIN_EPOCHS} Epochs)")
        
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.SWIN_EPOCHS, eta_min=1e-6)
        save_path = config.MODEL_CONFIGS['swin']['weights']
        
        best_loss = float('inf')
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(config.SWIN_EPOCHS):
            # Run Epoch
            train_loss, train_acc = self._run_epoch(optimizer, 'train')
            val_loss, val_acc = self._run_epoch(optimizer, 'val')
            
            # Store Metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc.cpu().item())
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc.cpu().item())

            # Log
            print(f"Epoch [{epoch+1}/{config.SWIN_EPOCHS}] "
                  f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                  f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Save Best
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"   >>> Val loss decreased to {val_loss:.4f}. Model Saved.")
            
            scheduler.step()
            
        return history

    def train_inception(self):
        print(f"\nSTARTING INCEPTION TRAINING (Phased)")
        
        save_path = config.MODEL_CONFIGS['inception']['weights']
        best_loss = float('inf') 
        full_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Internal Helper for Phases
        def run_phase(optimizer, epochs, phase_name, start_epoch_count):
            nonlocal best_loss
            print(f"\n{'-'*20} {phase_name} {'-'*20}")
            
            for epoch in range(epochs):
                # Run Epoch
                train_loss, train_acc = self._run_epoch(optimizer, 'train')
                val_loss, val_acc = self._run_epoch(optimizer, 'val')
                
                # Store Metrics
                full_history['train_loss'].append(train_loss)
                full_history['train_acc'].append(train_acc.cpu().item())
                full_history['val_loss'].append(val_loss)
                full_history['val_acc'].append(val_acc.cpu().item())
                
                # Log
                current_epoch = start_epoch_count + epoch + 1
                print(f"Epoch [{current_epoch}] "
                      f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                      f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
                
                # Save Best
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    print(f"   >>> Val loss decreased to {val_loss:.4f}. Model Saved.")
            
            return start_epoch_count + epochs

        # --- EXECUTE PHASES ---
        total_epochs_done = 0

        # Phase 1: Train Head Only
        print("Freezing base model...")
        for param in self.model.base_model.parameters(): param.requires_grad = False
        opt_p1 = optim.Adam(self.model.head.parameters(), lr=0.001)
        total_epochs_done = run_phase(opt_p1, config.INCEPTION_EPOCHS['phase1'], "PHASE 1: Head Only", total_epochs_done)

        # Phase 2: Fine-Tune Upper Blocks
        print("Unfreezing upper blocks...")
        for param in self.model.base_model.block8.parameters(): param.requires_grad = True
        for param in self.model.base_model.repeat_2.parameters(): param.requires_grad = True
        opt_p2 = optim.Adam(self.model.parameters(), lr=1e-5)
        total_epochs_done = run_phase(opt_p2, config.INCEPTION_EPOCHS['phase2'], "PHASE 2: Fine-Tuning", total_epochs_done)

        # Phase 3: Final Tuning (All Layers)
        print("Unfreezing all layers...")
        for param in self.model.parameters(): param.requires_grad = True
        opt_p3 = optim.Adam(self.model.parameters(), lr=1e-6)
        total_epochs_done = run_phase(opt_p3, config.INCEPTION_EPOCHS['phase3'], "PHASE 3: Final Polish", total_epochs_done)

        print(f"\nInception training completed. Best Val Loss: {best_loss:.4f}")
        return full_history

if __name__ == "__main__":
    TARGET_MODEL = 'inception'  # Options: 'swin', 'inception'

    try:
        trainer = Trainer(model_type=TARGET_MODEL)
        
        # 1. Start Training
        if TARGET_MODEL == 'swin':
            history = trainer.train_swin()
        else:
            history = trainer.train_inception()

        # 2. Save History to JSON
        json_path = os.path.join(config.BASE_DIR, f'history_{TARGET_MODEL}.json')
        with open(json_path, 'w') as f:
            json.dump(history, f)
        print(f"\nTraining history saved to: {json_path}")

        # 3. Generate Learning Curves Plot
        plot_path = os.path.join(config.BASE_DIR, f'learning_curves_{TARGET_MODEL}.png')
        utils.plot_learning_curves(history, plot_path)
        print(f"Plot saved to: {plot_path}")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")