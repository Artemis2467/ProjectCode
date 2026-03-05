from Functions import LogitConfig, LinearConfig, graph_roc_curve, count_files
from Layers import LogitModel, LinearModel
from DatasetLoader import logprob_test_loader, linear_test_loader

config = LogitConfig()
model = LogitModel(config)
length = count_files(r"models\logprob")
graph_roc_curve(
    config,
    logprob_test_loader,
    model,
    fr"logprob\{length}.pth",
)

config = LinearConfig()
model = LinearModel(config)
length = count_files(r"models\linear")
graph_roc_curve(
    config,
    linear_test_loader,
    model,
    fr"linear\{length}.pth",
)