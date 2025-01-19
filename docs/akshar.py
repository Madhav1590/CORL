import torch
import torch.nn as nn
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Model, GPT2Config

class HealthMonitoringTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, seq_len, num_classes):
        super(HealthMonitoringTransformer, self).__init__()
        config = GPT2Config(n_embd=hidden_dim, n_layer=num_layers, n_head=num_heads)
        self.transformer = GPT2Model(config)

        self.position_embedding = nn.Embedding(seq_len, hidden_dim)
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)

        self.anomaly_head = nn.Linear(hidden_dim, input_dim)
        self.personalization_head = nn.Linear(hidden_dim, 1)
        self.sentiment_head = nn.Linear(hidden_dim, 1)
        self.risk_assessment_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, task_type='anomaly'):
        position_ids = torch.arange(x.size(1), dtype=torch.long).unsqueeze(0).to(x.device)
        x = self.feature_embedding(x) + self.position_embedding(position_ids)

        transformer_outputs = self.transformer(inputs_embeds=x)
        hidden_states = transformer_outputs.last_hidden_state

        if task_type == 'anomaly':
            return self.anomaly_head(hidden_states)
        elif task_type == 'personalization':
            return self.personalization_head(hidden_states)
        elif task_type == 'sentiment':
            return self.sentiment_head(hidden_states)
        elif task_type == 'risk_assessment':
            return self.risk_assessment_head(hidden_states)



class HealthDataset(Dataset):
    def __init__(self, data, seq_len=50):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len, :-1]
        y = self.data[idx + 1: idx + self.seq_len + 1, :-1]  # shift target by 1
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


data = pd.read_csv('/home/mgoyani/CORL/docs/updated_data_with_height.csv')  # Assuming preprocessed and normalized CSV data
data = data.drop(columns=['heart_minutes', 'behavior_statement', 'behavior_status', 'gender'])
data = data.drop(columns=['user_id', 'start_time', 'stop_time'])  # Drop non-numeric fields
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(method='ffill').fillna(method='bfill')
data = data.values

# Create Dataset
seq_len = 50
dataset = HealthDataset(data, seq_len=seq_len)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=10, task_type='anomaly'):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        print(epoch)
        model.train()
        total_train_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x, task_type=task_type)


            if task_type == 'anomaly':
                loss = F.mse_loss(output, y)
            elif task_type == 'personalization':
                baseline = x.mean(dim=1, keepdim=True)
                deviation = torch.abs(output - baseline)
                loss = F.mse_loss(deviation, torch.zeros_like(deviation))
            elif task_type == 'sentiment':
                loss = F.binary_cross_entropy_with_logits(output, y)
            elif task_type == 'risk_assessment':
                loss = F.cross_entropy(output.view(-1, 2), y.view(-1).long())

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x, task_type=task_type)

                # Calculate validation loss
                if task_type == 'anomaly':
                    val_loss = F.mse_loss(output, y)
                elif task_type == 'personalization':
                    baseline = x.mean(dim=1, keepdim=True)
                    deviation = torch.abs(output - baseline)
                    val_loss = F.mse_loss(deviation, torch.zeros_like(deviation))
                elif task_type == 'sentiment':
                    val_loss = F.binary_cross_entropy_with_logits(output, y)
                elif task_type == 'risk_assessment':
                    val_loss = F.cross_entropy(output.view(-1, 2), y.view(-1).long())

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, test_loader, task_type='anomaly'):
    model.eval()
    total_test_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x, task_type=task_type)

            # Calculate test loss
            if task_type == 'anomaly':
                test_loss = F.mse_loss(output, y)
            elif task_type == 'personalization':
                baseline = x.mean(dim=1, keepdim=True)
                deviation = torch.abs(output - baseline)
                test_loss = F.mse_loss(deviation, torch.zeros_like(deviation))
            elif task_type == 'sentiment':
                test_loss = F.binary_cross_entropy_with_logits(output, y)
            elif task_type == 'risk_assessment':
                test_loss = F.cross_entropy(output.view(-1, 2), y.view(-1).long())

            total_test_loss += test_loss.item()

            # Store predictions and targets for evaluation metrics
            all_preds.append(output.cpu())
            all_targets.append(y.cpu())

    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

    # Additional evaluation metrics can be calculated based on task type, such as accuracy, F1 score, etc.
    return all_preds, all_targets


if __name__ == '__main__':
    device = torch.device("cuda")


    model = HealthMonitoringTransformer(input_dim=data.shape[1] - 1, hidden_dim=256, num_layers=4, seq_len=seq_len, num_classes=2, num_heads=8).to(device)

    train_model(model, train_loader, val_loader, epochs=100, task_type='anomaly')
    evaluate_model(model, test_loader, task_type='anomaly')