import random

from utils import *

options = Options()

# set all random seeds for reproducibility
torch.manual_seed(options.random_seed)
torch.cuda.manual_seed(options.random_seed)
random.seed(options.random_seed)

# set device
if options.device == "cuda" and torch.cuda.is_available():
    options.device = torch.device("cuda:0")
else:
    options.device = torch.device("cpu")

# Let's first prepare the MNIST dataset,
# run the test_dataset.py file to view some examples and see the dimensions of your tensor.
dataset = Dataset(options)

# define the model
model = VanillaAutoEncoder(options)
model.train(False)

# load the model parameters
load(model, options)

# display some images with its reconstruction
test_autoencoder(model, dataset, options)
generate_using_encoder(model, options)

plot_latent(model, dataset, options)
plot_latent_pca(model, dataset, options)
