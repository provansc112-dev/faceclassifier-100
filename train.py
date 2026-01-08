import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
import model as model_loader
from utils import get_dataloaders

class Trainer:
    def __init__(self, model_type):
        self.model_type = model_type
        self.device = config.DEVICE
        self.dataloaders, self.dataset_sizes = get_dataloaders(model_type)
        self.model = model_loader.load_trained_model(model_type)
        self.criterion = nn.CrossEntropyLoss()
        
    def _run_epoch(self, optimizer, phase, model_obj):
        model_obj.train() if phase == 'train' else model_obj.eval()
        running_loss, running_corrects = 0.0, 0
        
        pbar = tqdm(self.dataloaders[phase], desc=f"[{phase.upper()}]", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                if self.model_type == 'swin':
                    _, _, outputs = model_obj(images) # Swin return 3 values
                else:
                    outputs = model_obj(images) # Inception return logits
                
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels.long())

                if phase == 'train':
                    loss.backward()
                    if self.model_type == 'swin':
                        torch.nn.utils.clip_grad_norm_(model_obj.parameters(), max_norm=1.0)
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
        return epoch_loss, epoch_acc

    def train_swin(self, num_epochs=30):
        print(f"\nINITIALIZE TRAINING FOR SWIN TRANSFORMER ({num_epochs} Epoch) ---")
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        best_acc = 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = self._run_epoch(optimizer, 'train', self.model)
            val_loss, val_acc = self._run_epoch(optimizer, 'val', self.model)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "models/best_swin.pt")
                print("Saving Best Swin Model...")
            scheduler.step()

    def train_inception(self):
        # PHASE 1: Training Head Only
        print("\nINCEPTION PHASE 1: Training Head")
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        
        opt_p1 = optim.Adam(self.model.head.parameters(), lr=0.001)
        self._execute_phases(opt_p1, epochs=10, save_name="inc_p1.pth")

        # PHASE 2: Fine-Tuning
        print("\nINCEPTION PHASE 2: Fine-Tuning")
        for param in self.model.base_model.block8.parameters(): param.requires_grad = True
        for param in self.model.base_model.repeat_2.parameters(): param.requires_grad = True
        
        opt_p2 = optim.Adam(self.model.parameters(), lr=1e-5)
        self._execute_phases(opt_p2, epochs=10, save_name="inc_p2.pth")

        # PHASE 3: Final Tuning
        print("\nINCEPTION PHASE 3: Final Tuning")
        opt_p3 = optim.Adam(self.model.parameters(), lr=1e-6)
        self._execute_phases(opt_p3, epochs=5, save_name="inc_final.pth")

    def _execute_phases(self, optimizer, epochs, save_name):
        best_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(optimizer, 'train', self.model)
            val_loss, val_acc = self._run_epoch(optimizer, 'val', self.model)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f"models/{save_name}")
                print(f"New Best Val Acc: {best_acc:.4f} (Saved to {save_name})")

if __name__ == "__main__":
    m_type = 'swin' # or 'inception'
    task = Trainer(model_type=m_type)
    
    if m_type == 'swin':
        task.train_swin(num_epochs=config.NUM_EPOCHS)
    else:
        task.train_inception()