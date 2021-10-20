from typing import Tuple

import torch
import torch.nn as nn

from options import Options


class Print(nn.Module):
    """
    This model is for debugging purposes (place it in nn.Sequential to see tensor dimensions).
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x


class VanillaAutoEncoder(nn.Module):
    """
    Implementation of a default AutoEncoder.
    """

    def __init__(self, options: Options):
        super(VanillaAutoEncoder, self).__init__()
        self.encoder = Encoder(options)
        self.decoder = Decoder(options)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_vector = self.encoder(x)
        decoded_image = self.decoder(latent_vector)
        return decoded_image

    def generate(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_vector)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class VariationalAutoEncoder(nn.Module):
    """
    TODO: Here, your implementation of the Variational Autoencoder (VAE) should be made.
    Implement the forward method.
    """

    def __init__(self, options: Options):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = VariationalEncoder(options)
        self.decoder = Decoder(options)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :rtype: tuple consisting of (decoded image, latent_vector, mu, log_var)
        """
        pass

    def generate(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_vector)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)[0]


class Encoder(nn.Module):
    """
    The default Encoder network.
    """

    def __init__(self, options: Options):
        super(Encoder, self).__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, options.encoded_space_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    """
    The default Decoder network.
    """

    def __init__(self, options: Options):
        super(Decoder, self).__init__()
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(options.encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalEncoder(nn.Module):
    """
    The VAE uses the same decoder, but changes have to be made to the encoder.
    You can implement your changes below.
    """

    def __init__(self, options: Options):
        super(VariationalEncoder, self).__init__()
        """TODO: define your layers here"""

    def forward(self, x: torch.Tensor) -> tuple:
        """
        :rtype: tuple consisting of (latent vector, mu, log_var)
        """
        # from utils import reparameterize
        """
        TODO: Uncomment the line above
        Implement your forward method
        Keep yourself to the return type
        """


class Generator(nn.Module):
    """"
    TODO: Implement the generator of your GAN below.
    """
    pass


class Discriminator(nn.Module):
    """"
    TODO: Implement the discriminator of your GAN below.
    """
    pass
