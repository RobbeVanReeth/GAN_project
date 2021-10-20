import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from dataset import Dataset
from models import VanillaAutoEncoder, VariationalAutoEncoder
from options import Options


def train_autoencoder(model: VanillaAutoEncoder, options: Options, dataset: Dataset,
                      optimizer: torch.optim.Optimizer):
    distance = nn.MSELoss()

    for epoch in range(options.num_epochs):
        for data in dataset.train_loader:
            img, _ = data
            img = torch.Tensor(img).to(options.device)
            reconstruction = model(img)
            loss = distance(reconstruction, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, options.num_epochs, loss.item()))
        test_autoencoder(model, dataset, options)
        generate_using_encoder(model, options)


def train_vae(model: VariationalAutoEncoder, options: Options, dataset: Dataset,
              optimizer: torch.optim.Optimizer):
    """"
    TODO: This method should train your VAE, implement the code below.
    """
    pass


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """"
    You need to perform reparametrization for your VAE
    The goal of reparametrization is to have a probability involved to encode a value
    onto a certain place in the latent space.
    TODO: Implement this below.
    """
    pass


def save(model: nn.Module, options: Options):
    if not os.path.exists(options.save_path):
        os.makedirs(options.save_path)
    torch.save(model.state_dict(), options.save_path + options.model_name)


def load(model: nn.Module, options: Options):
    try:
        model.load_state_dict(torch.load(options.load_path + options.model_name))
        model.eval()
    except IOError:
        print("Could not load module!!")


def test_autoencoder(model: VanillaAutoEncoder, dataset: Dataset, options: Options):
    """"
    This method tests the autoencoder by plotting the original image and its reconstruction.
    """
    examples = enumerate(dataset.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(3):
        # plot reference
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

        # plot reconstructed
        plt.subplot(2, 3, i + 4)
        reconstruction = model.forward(example_data[i].unsqueeze(0).to(options.device))
        plt.tight_layout()
        plt.imshow(reconstruction.detach().squeeze(), cmap='gray', interpolation='none')
        plt.title("Reconstructed image: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig.show()


def test_vae(model: VariationalAutoEncoder, dataset: Dataset, options: Options):
    """"
    This method tests the VAE by plotting the original image and its reconstruction.
    """
    examples = enumerate(dataset.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(3):
        # plot reference
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

        # plot reconstructed
        plt.subplot(2, 3, i + 4)
        reconstruction, _, _, _ = model.forward(example_data[i].unsqueeze(0).to(options.device))
        plt.tight_layout()
        plt.imshow(reconstruction.detach().squeeze(), cmap='gray', interpolation='none')
        plt.title("Reconstructed image: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig.show()


def generate_using_encoder(model: nn.Module, options: Options):
    """"
    This method generates images using your module."""
    fig = plt.figure()
    for i in range(6):
        # plot generated images
        plt.subplot(2, 3, i + 1)
        gen_image = model.generate(torch.randn(1, options.encoded_space_dim).to(options.device))
        plt.tight_layout()
        plt.imshow(gen_image.detach().squeeze(), cmap='gray', interpolation='none')
        plt.title("Generated image: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    fig.show()


def plot_latent(autoencoder: nn.Module, dataset: Dataset, options: Options, num_batches: int = 100):
    """
    Plot the latent space to see how it differs between models.
    """
    for i, (x, y) in enumerate(dataset.test_loader):
        z = autoencoder.encode(x.to(options.device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()


def plot_latent_pca(autoencoder: nn.Module, dataset: Dataset, options: Options, num_batches: int = 100):
    """
    Plot the latent space to see how it differs between models.
    """
    pca = PCA(n_components=2)
    for i, (x, y) in enumerate(dataset.test_loader):
        z = autoencoder.encode(x.to(options.device))
        z = z.to('cpu').detach().numpy()
        pca.fit(z)
        reduced_z = pca.transform(z)
        plt.scatter(reduced_z[:, 0], reduced_z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()


def plot_latent_pca_3d(autoencoder: nn.Module, dataset: Dataset, options: Options, num_batches: int = 100):
    """
    Plot the latent space to see how it differs between models.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pca = PCA(n_components=3)
    for i, (x, y) in enumerate(dataset.test_loader):
        z = autoencoder.encode(x.to(options.device))
        z = z.to('cpu').detach().numpy()
        pca.fit(z)
        reduced_z = pca.transform(z)
        ax.scatter(reduced_z[:, 0], reduced_z[:, 1], reduced_z[:, 2], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()
