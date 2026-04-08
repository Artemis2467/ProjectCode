import os
import json
from LinearTrain import linear_train
from LogprobTrain import logprob_train
from Functions import LogitConfig, LinearConfig, test_model, plot_loss, calculate_threshold
from DatasetLoader import logprob_test_loader, linear_test_loader, logprob_val_loader, linear_val_loader
from Layers import LogitModel, LinearModel

def logprob_train_model(config, model, count, weighted):
    history = logprob_train(config, f"{"weighted" if weighted else ""}{count}.pth")
    threshold = calculate_threshold(model, config, logprob_val_loader)
    auroc, f1, fpr, tpr, report, report_dict = test_model(config, logprob_test_loader, threshold, model, fr"logprob\{"weighted" if weighted else ""}{count}.pth")

    print(f"\n{report}\nauroc: {auroc}")

    with open(r"history\logprob_history.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{count}\n dimension: {config.d_model} lr: {config.learning_rate}{f" CNN channel: {config.conv_ch}" if config.add_conv else ""} auroc: {auroc} f1: {f1}")

    if auroc >= 0.60 and f1 >= 0.70:
        plot_loss(history, is_linear=False, show=False, length=f"{"weighted" if weighted else ""}{count}")

        report_dict = report_dict["1.0"]
        accuracy = report_dict["accuracy"]
        del report_dict["support"]
        report_dict["ID"] = f"{"weighted" if weighted else ""}{count}"
        report_dict["accuracy"] = accuracy
        report_dict["auroc"] = auroc
        report_dict["d_model"] = config.d_model
        report_dict["learning_rate"] = config.learning_rate
        report_dict["conv_ch"] = config.conv_ch if config.add_conv else None

        with open(r"history\logprob_best.jsonl", "a", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False)
            f.write("\n")
    
    else:
        os.remove(fr"models\logprob\sec_try{count}.pth")

def linear_train_model(config, model, count, weighted):
    history = linear_train(config, f"{"weighted" if weighted else ""}{count}.pth")
    threshold = calculate_threshold(model, config, linear_val_loader)
    auroc, f1, fpr, tpr, report, report_dict = test_model(config, linear_test_loader, threshold, model, fr"linear\{"weighted" if weighted else ""}{count}.pth")
    print(f"\n{report}\nauroc: {auroc}")

    with open(r"history\linear_history.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{count}\n dimension: {config.d_model} lr: {config.learning_rate} auroc: {auroc} f1: {f1}")

    if auroc >= 0.60 and f1 >= 0.70:
        plot_loss(history, is_linear=True, show=False, length=f"{"weighted" if weighted else ""}{count}")
        
        report_dict = report_dict["1.0"]
        del report_dict["support"]
        report_dict["ID"] = f"{"weighted" if weighted else ""}{count}"
        report_dict["auroc"] = auroc
        report_dict["d_model"] = config.d_model
        report_dict["learning_rate"] = config.learning_rate

        with open(r"history\linear_best.jsonl", "a", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False)
            f.write("\n")
    
    else:
        os.remove(fr"models\linear\sec_try{count}.pth")

if __name__ == "__main__":
    while True:
        type = input("linear or logporb --> ")
        weighted = input("weighted? [y/n] ") == "y"

        if type == "linear":
            config = LinearConfig()
            count = 1
            total = len(config.d_model_choices) * len (config.learning_rate_choices)

            for dimension in config.d_model_choices:
                config.d_model = dimension
                for lr in config.learning_rate_choices:
                    config.learning_rate = lr
                    print(f"{count} / {total}\n")
                    print(f"model dimension: {config.d_model} learning rate: {config.learning_rate}")
                    model = LinearModel(config)
                    linear_train_model(config, model, count, weighted)
                    count += 1
            break

        elif type == "logprob":
            config = LogitConfig()
            count = 1
            total = len(config.d_model_choices) * len(config.learning_rate_choices) * (1 + len(config.conv_ch_choices) if len(config.add_conv_choices) == 2 else 1)

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
                                
                                print(f"{count} / {total}\n")
                                print(f"model dimension: {config.d_model} learning rate: {config.learning_rate} CNN channel: {config.conv_ch}")
                                model = LogitModel(config)
                                logprob_train_model(config, model, count, weighted)
                                count += 1

                        else:
                            print(f"{count} / {total}")
                            print(f"model dimension: {config.d_model} learning rate: {config.learning_rate}")
                            model = LogitModel(config)
                            logprob_train_model(config, model, count, weighted)
                            count += 1
            break

        else:
            print("\n---------input [linear] or [logprob]----------\n")

