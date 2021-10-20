import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

from options import Options


class Dataset:
    def __init__(self, options: Options):
        """
        Note: options.dataset should contain either MNIST or CIFAR10.
        """
        self.options = options

        if options.dataset == "CIFAR10":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                                                    transform=transform)
            testset = torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                                   transform=transform)
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
            trainset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                  transform=transform)
            testset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                                 transform=transform)

        if options.dataset not in ["MNIST", "CIFAR10"]:
            print("Dataset not correctly defined, loading the MNIST digits dataset...")

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=options.batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=options.batch_size_test, shuffle=True)

    def show_examples(self):
        """
        Show some examples of the selected dataset.
        """
        examples = enumerate(self.test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        print(f"The shape of the training data tensor is: {example_data.shape}")

        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            if self.options.dataset == "CIFAR10":
                example_data[i] = example_data[i] / 2 + 0.5  # unnormalize
                to_img = transforms.ToPILImage(mode="RGB")
                img = to_img(example_data[i])
                plt.imshow(img)
            else:
                plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        fig.show()
