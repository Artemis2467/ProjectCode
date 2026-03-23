import torch
import torch.optim as optim
from Layers import LinearModel
from DatasetLoader import linear_train_loader, linear_val_loader, linear_test_loader
from Functions import LinearConfig, plot_loss, run_batch, test_model, graph_roc_curve

def linear_train(config, model_pth):

    model = LinearModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    prev_loss = float('inf')
    history = {'running_loss': [], 'val_loss': []}

    for epoch in range(config.num_epochs):
        
        running_loss = run_batch(config=config,
                loader=linear_train_loader,
                model=model,
                optimizer=optimizer,
                type="train",
                epoch=epoch
                )

        val_loss = run_batch(config=config,
                            loader=linear_val_loader,
                            model=model,
                            optimizer=optimizer,
                            type="val",
                            )
        
        running_loss /= len(linear_train_loader.dataset)
        val_loss /= len(linear_val_loader.dataset)
        history['running_loss'].append(running_loss)
        history['val_loss'].append(val_loss)
                
        if val_loss < prev_loss:
            prev_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), fr'models\linear\{model_pth}')
        else:
            patience_count += 1
            if patience_count >= config.stop_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    return history


if __name__ == "__main__":
    model_pth = input("model's parameter path: ")

    config = LinearConfig()
    history = linear_train(config, model_pth)
    model = LinearModel(config)

    plot_loss(history, is_linear=True, show=True)

    auroc, f1, fpr, tpr = test_model(config, linear_test_loader, model, fr"linear\{model_pth}")
    graph_roc_curve(auroc, f1, fpr, tpr, is_linear=True)