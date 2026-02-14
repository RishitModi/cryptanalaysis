import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path="/content/speck32_traces_88_10k.h5"
with h5py.File(file_path, "r") as f:
    traces=np.array(f["traces"], dtype=np.float32)
    pts=np.array(f["plaintexts"], dtype=np.uint32)
    key=int(f["key"][0])


labels=np.full(len(traces), key, dtype=np.uint8)

mean=np.mean(traces)
std=np.std(traces)
traces=(traces - mean) / std

noise_level=0.05
noise=np.random.normal(0, noise_level, traces.shape)
traces=traces+noise

def desynchronize(traces, max_shift=5):
    desynced=np.zeros_like(traces)
    for i in range(len(traces)):
        shift=np.random.randint(-max_shift, max_shift + 1)
        desynced[i]=np.roll(traces[i], shift)
    return desynced

traces=desynchronize(traces, max_shift=5)



X_tensor=torch.tensor(traces, dtype=torch.float32)
y_tensor=torch.tensor(labels, dtype=torch.long)

dataset=TensorDataset(X_tensor, y_tensor)
train_size=int(0.8 * len(dataset))
test_size=len(dataset) - train_size
train_data, test_data=random_split(dataset, [train_size, test_size])
train_loader=DataLoader(train_data, batch_size=64, shuffle=True)
test_loader=DataLoader(test_data, batch_size=64, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1=nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool=nn.MaxPool1d(2)
        self.conv2=nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1=nn.Linear(32 * 22, 64)
        self.fc2=nn.Linear(64, 256) 
    def forward(self, x):
        x=x.unsqueeze(1)
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

model=CNNModel().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)

epochs = 25
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss=criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

model.eval()
all_outputs=[]
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch=X_batch.to(device)
        outputs=model(X_batch)
        all_outputs.append(outputs.cpu())
outputs=torch.cat(all_outputs, dim=0)
log_probs=F.log_softmax(outputs, dim=1).numpy()

def compute_rank(log_probs, true_key):
    log_likelihood = np.sum(log_probs, axis=0)
    sorted_idx = np.argsort(log_likelihood)[::-1]
    rank = np.where(sorted_idx == true_key)[0][0]
    return rank

rank = compute_rank(log_probs, key)
print(f"\nBest subkey guess: {np.argmax(np.sum(log_probs, axis=0))}")
print(f"True subkey: {key}")
print(f"Key rank: {rank}")
