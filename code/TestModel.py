import os
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from Functions import run_batch, LogitConfig, LinearConfig
from Layers import LogitModel, LinearModel
from DatasetLoader import logprob_test_loader, linear_test_loader

def graph_roc_curve(config, test_loader, model, parameter_path: str):

    model.load_state_dict(torch.load(os.path.join("models", parameter_path)))

    results, targets = run_batch(
        config,
        test_loader,
        model,
        None,
        "test",
        None
    )

    fpr, tpr, threshold = roc_curve(targets, results)
    auroc = auc(fpr, tpr)

    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {"linear model" if model.is_linear else "logprob model"}')
    plt.legend()
    plt.show()

config = LogitConfig()
model = LogitModel(config)

graph_roc_curve(
    config,
    logprob_test_loader,
    model,
    "best_logprob_model.pth"
)

config = LinearConfig()
model = LinearModel(config)
graph_roc_curve(
    config,
    linear_test_loader,
    model,
    "best_linear_model.pth"
)