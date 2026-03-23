import os
from LinearTrain import linear_train
from LogprobTrain import logprob_train
from Functions import LogitConfig, LinearConfig, test_model, count_files, plot_loss, graph_roc_curve
from DatasetLoader import logprob_test_loader, linear_test_loader
from Layers import LogitModel, LinearModel

def logprob_train_model(config, model, count):
    length = count_files(r"models\logprob")
    history = logprob_train(config, f"{length + 1}.pth")
    auroc, f1, fpr, tpr = test_model(config, logprob_test_loader, model, fr"logprob\{length + 1}.pth")
    print(f"auroc: {auroc}, f1: {f1}")
    with open(r"history\logprob_history.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{count}\n dimension: {config.d_model} lr: {config.learning_rate}{f" CNN channel: {config.conv_ch}" if config.add_conv else ""} auroc: {auroc} f1: {f1}")

    if auroc >= 0.60:
        plot_loss(history, is_linear=False, show=False)
        graph_roc_curve(
            auroc=auroc,
            f1 = f1,
            fpr=fpr,
            tpr=tpr,
            is_linear=False
        )

        with open(r"history\logprob_best.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{length + 1}\n dimension: {config.d_model} lr: {config.learning_rate}{f" CNN channel: {config.conv_ch}" if config.add_conv else ""}")
    
    else:
        os.remove(fr"models\logprob\{length + 1}.pth")

def linear_train_model(config, model, count):
    length = count_files(r"models\linear")
    history = linear_train(config, f"{length + 1}.pth")
    auroc, f1, fpr, tpr = test_model(config, linear_test_loader, model, fr"linear\{length + 1}.pth")
    print(f"auroc: {auroc}, f1: {f1}")
    with open(r"history\linear_history.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{count}\n dimension: {config.d_model} lr: {config.learning_rate} auroc: {auroc} f1: {f1}")

    if auroc >= 0.60:
        plot_loss(history, is_linear=True, show=False)
        graph_roc_curve(
            auroc=auroc,
            f1 = f1,
            fpr=fpr,
            tpr=tpr,
            is_linear=True
        )

        with open(r"history\linear_best.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{length + 1}\n dimension: {config.d_model} lr: {config.learning_rate}")
    
    else:
        os.remove(fr"models\linear\{length + 1}.pth")

if __name__ == "__main__":
    while True:
        type = input("linear or logporb --> ")

        if type == "linear":
            config = LinearConfig()
            count = 1

            for dimension in config.d_model_choices:
                config.d_model = dimension
                for lr in config.learning_rate_choices:
                    config.learning_rate = lr
                    print(f"{count} / 20\n")
                    print(f"model dimension: {config.d_model} learning rate: {config.learning_rate}")
                    model = LinearModel(config)
                    linear_train_model(config, model, count)
                    count += 1
            break



        elif type == "logprob":
            config = LogitConfig()
            count = 1

            for is_conv in config.add_conv_choices:
                print("----With CNN layer----\n" if config.add_conv else "----No CNN layer----\n")
                config.add_conv = is_conv
                for item in config.d_model_choices:
                    config.d_model = item
                    for lr in config.learning_rate_choices:
                        config.learning_rate = lr
                        if config.add_conv:
                            for i in config.conv_ch_choices:
                                config.conv_ch = i
                                
                                print(f"{count} / 60\n")
                                print(f"model dimension: {config.d_model} learning rate: {config.learning_rate} CNN channel: {config.conv_ch}")
                                model = LogitModel(config)
                                logprob_train_model(config, model, count)
                                count += 1

                        else:
                            print(f"{count} / 60")
                            print(f"model dimension: {config.d_model} learning rate: {config.learning_rate}")
                            model = LogitModel(config)
                            logprob_train_model(config, model, count)
                            count += 1
            break

        else:
            print("\n---------input [linear] or [logprob]----------\n")

