import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from Layers import LogitModel
from Functions import LogitConfig, plot_loss
from DatasetLoader import logprob_train_loader, logprob_val_loader

config = LogitConfig()
device = config.device

model = LogitModel(config)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

history = {"running_loss": [], "val_loss": []}
prev_loss = float("inf")

for epoch in range(config.num_epochs):
    model.train()

    running_loss = 0.0
    for batch in tqdm(logprob_train_loader, desc=f"epoch: {config.num_epochs}/{epoch + 1}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(
            batch["logprobs1"],
            batch["logprobs2"],
            batch["cos_sim"],
            batch["entropy"]
        )
        loss = criterion(output, batch["labels"].unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch["cos_sim"].size(0)

    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for batch in logprob_val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(
                batch["logprobs1"],
                batch["logprobs2"],
                batch["cos_sim"],
                batch["entropy"]
            )
            loss = criterion(output, batch["labels"].unsqueeze(1))

            val_loss += loss.item() * batch["cos_sim"].size(0)

    running_loss /= len(logprob_train_loader.dataset)
    val_loss /= len(logprob_val_loader.dataset)
    history["running_loss"].append(running_loss)
    history["val_loss"].append(val_loss)

    if val_loss < prev_loss:
        prev_loss = val_loss
        patience_count = 0
        torch.save(model.state_dict(), 'models/best_logprob_model.pth')
    else:
        patience_count += 1
        if patience_count >= config.patience:
            print(f"model stopped at {epoch} epochs.")
            break

plot_loss(history, is_linear=False)