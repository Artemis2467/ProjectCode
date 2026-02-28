import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Layers import LinearModel
from DatasetLoader import linear_train_loader, linear_val_loader
from Functions import LinearConfig
import matplotlib.pyplot as plt

config = LinearConfig()
device = config.device
model = LinearModel(config)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_loss = float('inf')
no_improve_epoch = 0
history = {'train_loss': [], 'val_loss': []}

for epoch in range(config.num_epochs):
    
    model.train()

    running_loss = 0.0
    for batch in tqdm(linear_train_loader, desc=f"{epoch + 1}/{config.num_epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(batch["cos_sim"], batch["entropy"])
        loss = criterion(output, batch["label"].unsqueeze(1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * batch["cos_sim"].size(0)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in linear_val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch["cos_sim"], batch["entropy"])
            loss = criterion(output, batch["label"].unsqueeze(1))
            val_loss += loss.item() * batch["cos_sim"].size(0)
            
        scheduler.step(val_loss)
    
    running_loss /= len(linear_train_loader)
    val_loss /= len(linear_val_loader)
    history['train_loss'].append(running_loss)
    history['val_loss'].append(val_loss)
            
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epoch = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve_epoch += 1
        if no_improve_epoch >= config.stop_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        print(f'Epoch {epoch + 1}: Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}')


plt.figure(figsize=(12, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()
plt.show()
