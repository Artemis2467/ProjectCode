import matplotlib.pyplot as plt
from Functions import test_model, calculate_threshold, LogitConfig, LinearConfig
from Layers import LogitModel, LinearModel
from DatasetLoader import logprob_test_loader, linear_test_loader, logprob_val_loader, linear_val_loader
from StoreDataset import FileLoader

COLORS = ["b", "g", "r", "c", "m", "y", "k"]

if __name__ == "__main__":
    mode = input("logprob or linear or both --> ")

    if mode == "logprob":

        compare = False
        color = 0

        plt.figure(1)
        plt.plot([0, 1], color="grey", linestyle="dashed", label="random model")

        while True:

            #initialize configuration
            config = LogitConfig()
            
            #asks for file location
            id = int(input("ID of model to test -->"))
            weighted = input("weighted? [y/n] ") == "y"
            location = f"{"weighted" if weighted else ""}{id}"
            file = FileLoader(rf"..\history\logprob_best.jsonl")

            #checks for ID
            parameters = None
            for item in file:
                if item["ID"] == location:
                    parameters = item
                    break
            if not parameters:
                raise RuntimeError("ID not in file")
            
            # changes config for testing
            if parameters["conv_ch"]:
                config.add_conv = True
                config.conv_ch = parameters["conv_ch"]
            else:
                config.add_conv = False
            
            config.d_model = parameters["d_model"]
            
            # Initialize model
            model = LogitModel(config)

            plot_distribution = input("Plot model distribution: [y/n]") == "y"

            # calculate f1 score and auroc score
            threshold = calculate_threshold(model, config, logprob_val_loader)
            auroc, f1, fpr, tpr, report, report_dict = test_model(
                config, 
                logprob_test_loader, 
                threshold, 
                model, 
                parameter_path=fr"logprob\{location}.pth",
                plot_distribution=plot_distribution
                )


            print(report)
            print(f"threshold: {threshold}")
            print(f"auroc score: {auroc}")

            if plot_distribution:
                break

            label = input("model label: ")
            plt.plot(fpr, tpr, label=f'{label} (AUC=%0.2f)' % auroc, color=f"{COLORS[color]}")

            # asks if user want to compare between logprob models
            compare = input("Compare between models? [y/n] ") == "y"
            color += 1
            if color >= 7:
                color = 0

            if not compare:
                break

        if not plot_distribution:   
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves')
            plt.legend()
            save = input("save? [y/n] ") == "y"

            if save:
                name = input("input name: ")
                plt.savefig(fr"test_results\ROC\logprob\{name}.png")

            plt.show()


    elif mode == "linear":
        color = 0
        compare = False

        plt.figure(1)
        plt.plot([0, 1], color="grey", linestyle="dashed", label="random model")

        while True:
            config = LinearConfig()
            id = int(input("ID of model to test -->"))
            weighted = input("weighted? [y/n] ") == "y"
            location = f"{"weighted" if weighted else ""}{id}"
 
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


            plot_distribution = input("Plot model distribution: [y/n]") == "y"

            threshold = calculate_threshold(model, config, linear_val_loader)
            auroc, f1, fpr, tpr, report, report_dict = test_model(
                config, 
                linear_test_loader, 
                threshold, 
                model, 
                parameter_path=fr"linear\{location}.pth",
                plot_distribution=plot_distribution
                )
            
            print(report)
            print(f"threshold: {threshold}")
            print(f"auroc score: {auroc}")
            
            if plot_distribution:
                break
            
            label = input("model label: ")
            plt.plot(fpr, tpr, label=f'{label} (AUC=%0.2f)' % auroc, color=COLORS[color])

            compare = input("Compare between models? [y/n] ") == "y"
            color += 1
            if color >= 7:
                color = 0

            if not compare:
                break
        if not plot_distribution: 
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves')
            plt.legend()

            save = input("save? [y/n] ") == "y"

            if save:
                name = input("input name: ")
                plt.savefig(fr"test_results\ROC\linear\{name}.png")

            plt.show()


    elif mode == "both":

        color = 0
        compare = False
        plt.figure()
        plt.plot([0, 1], color="grey", linestyle="dashed", label="random model")

        while True:

            config = LogitConfig()
            
            id = int(input("ID of logprob model to test -->"))
            weighted = input("weighted? [y/n] ") == "y"
            location = f"{"weighted" if weighted else ""}{id}"
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

            auroc, f1, fpr, tpr, report, report_dict = test_model(config, logprob_test_loader, 0.5, model, parameter_path=fr"logprob\{location}.pth")
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
            
            id = int(input("ID of linear model to test -->"))
            weighted = input("weighted? [y/n] ") == "y"
            location = f"{"weighted" if weighted else ""}{id}"
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

            auroc, f1, fpr, tpr, report, report_dict = test_model(config, linear_test_loader, 0.5, model, parameter_path=fr"linear\{location}.pth")
            label = input("model label: ")
            plt.plot(fpr, tpr, label=f'{label} (AUC=%0.2f)' % auroc, color=f"{COLORS[color]}")

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

        save = input("save? [y/n] ") == "y"

        if save:
            name = input("input name: ")
            plt.savefig(fr"test_results\ROC\{name}.png")
        plt.show()

    else:
        raise TypeError("Mode not logprob or linear or all")
