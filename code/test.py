import torch
import os
import matplotlib.pyplot as plt
from Functions import test_model, LogitConfig, LinearConfig
from Layers import LogitModel, LinearModel
from DatasetLoader import logprob_test_loader, linear_test_loader
from StoreDataset import FileLoader

COLORS = ["b", "g", "r", "c", "m", "y", "k"]

if __name__ == "__main__":
    mode = input("logprob or linear or both --> ")

    if mode == "logprob":

        compare = False
        color = 0

        plt.figure()
        plt.plot([0, 1], color="grey", linestyle="dashed", label="random model")

        while True:

            config = LogitConfig()
            
            location = int(input("ID of model to test -->"))
            file = FileLoader(rf"..\history\logprob_best.jsonl")

            parameters = None
            for item in file:
                if item["ID"] == location:
                    parameters = item
                    break
            if not parameters:
                raise RuntimeError("ID not in file")
            
            if parameters["conv_ch"]:
                config.add_conv = True
                config.conv_ch = parameters["conv_ch"]
            else:
                config.add_conv = False
            
            config.d_model = parameters["d_model"]
            
            model = LogitModel(config)

            auroc, f1, fpr, tpr, report, report_dict = test_model(config, logprob_test_loader, model, parameter_path=fr"logprob\sec_try{location}.pth")
            label = input("model label: ")
            plt.plot(fpr, tpr, label=f'{label} (AUC=%0.2f)' % auroc, color=f"{COLORS[color]}")

            print(report)
            print(auroc)

            compare = input("Compare between models? [y/n] ") == "y"
            color += 1
            if color >= 7:
                color = 0

            if not compare:
                break
              
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves')
        plt.legend()
        plt.show()

    elif mode == "linear":
        
        compare = False

        plt.figure()
        plt.plot([0, 1], color="grey", linestyle="dashed", label="random model")

        while True:
            config = LinearConfig()
            
            location = int(input("ID of model to test -->"))
            file = FileLoader(rf"..\history\linear_best.jsonl")

            parameters = None
            for item in file:
                if item["ID"] == location:
                    parameters = item
                    break
            if not parameters:
                raise RuntimeError("ID not in file")
            
            config.d_model = parameters["d_model"]
            
            model = LinearModel(config)

            auroc, f1, fpr, tpr, report, report_dict = test_model(config, linear_test_loader, model, parameter_path=fr"linear\sec_try{location}.pth")
            label = input("model label: ")
            plt.plot(fpr, tpr, label=f'{label} (AUC=%0.2f)' % auroc)

            print(report)
            print(auroc)

            compare = input("Compare between models? [y/n] ") == "y"

            if not compare:
                break

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves')
        plt.legend()
        plt.show()


    elif mode == "both":

        color = 0
        compare = False
        plt.figure()
        plt.plot([0, 1], color="grey", linestyle="dashed", label="random model")

        while True:

            config = LogitConfig()
            
            location = int(input("ID of model to test -->"))
            file = FileLoader(rf"..\history\logprob_best.jsonl")

            parameters = None
            for item in file:
                if item["ID"] == location:
                    parameters = item
                    break
            if not parameters:
                raise RuntimeError("ID not in file")
            
            if parameters["conv_ch"]:
                config.add_conv = True
                config.conv_ch = parameters["conv_ch"]
            else:
                config.add_conv = False
            
            config.d_model = parameters["d_model"]
            
            model = LogitModel(config)

            auroc, f1, fpr, tpr, report, report_dict = test_model(config, logprob_test_loader, model, parameter_path=fr"logprob\sec_try{location}.pth")
            label = input("model label: ")
            plt.plot(fpr, tpr, label=f'{label} (AUC=%0.2f)' % auroc, color=f"{COLORS[color]}")

            print(report)
            print(auroc)

            compare = input("Compare between logprob models? [y/n] ") == "y"
            color += 1
            if color >= 7:
                color = 0

            if not compare:
                break

        while True:
            config = LinearConfig()
            
            location = int(input("ID of model to test for hybrid model -->"))
            file = FileLoader(rf"..\history\linear_best.jsonl")

            parameters = None
            for item in file:
                if item["ID"] == location:
                    parameters = item
                    break
            if not parameters:
                raise RuntimeError("ID not in file")
            
            config.d_model = parameters["d_model"]
            
            model = LinearModel(config)

            auroc, f1, fpr, tpr, report, report_dict = test_model(config, linear_test_loader, model, parameter_path=fr"linear\sec_try{location}.pth")
            label = input("model label: ")
            plt.plot(fpr, tpr, label=f'{label} (AUC=%0.2f)' % auroc)

            print(report)
            print(auroc)

            compare = input("Compare between models? [y/n] ") == "y"

            if not compare:
                break

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves')
        plt.legend()
        plt.show()

    else:
        raise TypeError("Mode not logprob or linear or all")

