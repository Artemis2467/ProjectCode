import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Layers import LogitModel
from Functions import LogitConfig, plot_loss, run_batch, test_model, graph_roc_curve
from DatasetLoader import logprob_train_loader, logprob_val_loader, logprob_test_loader

def logprob_train(config, model_pth):

    model = LogitModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=5,
        factor=0.5
    )

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

        scheduler.step(val_loss)

        if val_loss < prev_loss:
            prev_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), fr'models\logprob\{model_pth}')
        else:
            patience_count += 1
            if patience_count >= config.patience:
                print(f"model stopped at {epoch} epochs.")
                break

    return history

if __name__ == "__main__":
    model_pth = input("model's parameter path: ")

    config = LogitConfig()
    history = logprob_train(config, model_pth)
    model = LogitModel(config)

    plot_loss(history, is_linear=False, show=True)

    auroc, f1, fpr, tpr = test_model(config, logprob_test_loader, model, fr"logprob\{model_pth}")
    graph_roc_curve(auroc, f1, fpr, tpr, is_linear=False)

    