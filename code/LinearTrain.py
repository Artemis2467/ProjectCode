import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Layers import LinearModel
from DatasetLoader import linear_train_loader, linear_val_loader
from Functions import LinearConfig, plot_loss

config = LinearConfig()
device = config.device
model = LinearModel(config)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

prev_loss = float('inf')
history = {'running_loss': [], 'val_loss': []}

for epoch in range(config.num_epochs):
    
    model.train()

    running_loss = 0.0
    for batch in tqdm(linear_train_loader, desc=f"epoch: {config.num_epochs}/{epoch + 1} validation loss: {prev_loss} "):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(batch["cos_sim"], batch["entropy"])
        loss = criterion(output, batch["label"].unsqueeze(1))
        loss.backward()
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
    
    running_loss /= len(linear_train_loader)
    val_loss /= len(linear_val_loader)
    history['running_loss'].append(running_loss)
    history['val_loss'].append(val_loss)
            
    if val_loss < prev_loss:
        prev_loss = val_loss
        patience_count = 0
        torch.save(model.state_dict(), 'models/best_linear_model.pth')
    else:
        patience_count += 1
        if patience_count >= config.stop_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

plot_loss(history)
