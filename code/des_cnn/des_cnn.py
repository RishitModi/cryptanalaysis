import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.utils.data import random_split

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

with h5py.File("/content/des_traces_with_metadata.h5", "r") as f:
    X=f["traces"][:]
    ciphertexts=f["ciphertexts"][:]
    plaintexts=f["plaintexts"][:]
    keys=f["keys"][:]


X=(X-np.mean(X, axis=1, keepdims=True))/np.std(X, axis=1, keepdims=True)

SBOX1=np.array([
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
    [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
    [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
    [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
])

def sbox_output(ptext_byte, key_byte):
    expanded=(int(ptext_byte)^int(key_byte))&0b111111
    row=((expanded & 0b100000)>>4)|(expanded & 0b000001)
    col=(expanded & 0b011110)>>1
    return int(SBOX1[row, col])

labels_list=[]

for i in range(len(plaintexts)):
    ptext_byte=plaintexts[i][0]
    key_byte=keys[i][0]
    sbox_val=sbox_output(ptext_byte, key_byte)
    labels_list.append(sbox_val)

labels=np.array(labels_list, dtype=np.int64)

X_tensor=torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor=torch.tensor(labels, dtype=torch.long)

train_size=int(0.8 * len(X_tensor))
test_size=len(X_tensor) - train_size
train_dataset, test_dataset=random_split(TensorDataset(X_tensor, y_tensor), [train_size, test_size])

train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
test_loader=DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)


class DES_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv1d(1, 32, kernel_size=11, padding=5)
        self.conv2=nn.Conv1d(32, 128, kernel_size=11, padding=5)
        self.conv3=nn.Conv1d(128, 256, kernel_size=11, padding=5)
        self.pool=nn.AvgPool1d(2)
        self.global_pool=nn.AdaptiveAvgPool1d(1)
        self.fc1=nn.Linear(256, 64)
        self.fc2=nn.Linear(64, 16)

    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=F.relu(self.conv3(x))
        x=self.pool(x)
        x=self.global_pool(x)
        x=x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        return self.fc2(x)

model=DES_CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=1e-4)

num_epochs=25
for epoch in range(num_epochs):
    model.train()
    total_loss=0
    total_correct=0
    total_samples=0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs=model(X_batch)
        loss=criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()*X_batch.size(0)
        total_correct+=(outputs.argmax(dim=1)==y_batch).sum().item()
        total_samples+=X_batch.size(0)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss:{total_loss/total_samples:.4f}  Acc: {total_correct/total_samples:.4f}")


model.eval()
all_outputs=[]
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch=X_batch.to(device)
        outputs=model(X_batch)
        all_outputs.append(outputs.cpu())
outputs=torch.cat(all_outputs, dim=0)
log_probs=F.log_softmax(outputs, dim=1).numpy()


num_subkeys=64
scores=np.zeros(num_subkeys)

test_indices=test_dataset.indices
plaintexts_test=plaintexts[test_indices]
keys_test=keys[test_indices]

for guess in range(num_subkeys):
    total=0.0
    for i in range(len(plaintexts_test)):
        label=sbox_output(plaintexts_test[i][0] & 0b111111, guess)
        total+=log_probs[i][label]
    scores[guess]=total

best_guess=np.argmax(scores)

true_subkey=int(keys[0][0])&0b111111

sorted_indices=np.argsort(scores)[::-1]
print(sorted_indices)
rank=np.where(sorted_indices == true_subkey)[0][0]

print("Best subkey guess:", best_guess)
print("True subkey:", true_subkey)
print("Key rank:", rank)

