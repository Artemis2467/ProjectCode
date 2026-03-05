from Functions import LogitConfig, LinearConfig, graph_roc_curve
from Layers import LogitModel, LinearModel
from DatasetLoader import logprob_test_loader, linear_test_loader

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