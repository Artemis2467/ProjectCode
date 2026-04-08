from Functions import calculate_threshold, LinearConfig
from Layers import LinearModel
from DatasetLoader import linear_val_loader

config = LinearConfig()
model = LinearModel(config)

thresholds = []


for i in range(50):
    threshold = calculate_threshold(
        model=model,
        config=config,
        val_loader=linear_val_loader
    )

    thresholds.append(threshold)



