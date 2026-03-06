import torch
import torch.optim as optim
from Layers import LogitModel
from Functions import LogitConfig, plot_loss, run_batch, count_files
from DatasetLoader import logprob_train_loader, logprob_val_loader

config = LogitConfig()
device = config.device
length = count_files(r"models\logprob")

model = LogitModel(config)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

history = {"running_loss": [], "val_loss": []}
prev_loss = float("inf")

for epoch in range(config.num_epochs):
    model.train()

    running_loss = run_batch(
        config=config,
        loader=logprob_train_loader,
        model=model,
        optimizer=optimizer,
        type="train",
        epoch=epoch
    )

    val_loss = run_batch(
        config=config,
        loader=logprob_val_loader,
        model=model,
        optimizer=optimizer,
        type="val",
    )

    running_loss /= len(logprob_train_loader.dataset)
    val_loss /= len(logprob_val_loader.dataset)
    history["running_loss"].append(running_loss)
    history["val_loss"].append(val_loss)

    if val_loss < prev_loss:
        prev_loss = val_loss
        patience_count = 0
        torch.save(model.state_dict(), fr'models\logprob\{length + 1}.pth')
    else:
        patience_count += 1
        if patience_count >= config.patience:
            print(f"model stopped at {epoch} epochs.")
            break

plot_loss(history, is_linear=False)