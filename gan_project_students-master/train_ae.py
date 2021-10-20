import random

from utils import *

options = Options()

# set all random seeds for reproducibility
torch.manual_seed(options.random_seed)
torch.cuda.manual_seed(options.random_seed)
random.seed(options.random_seed)

# set device
print(f'Is cuda available: {torch.cuda.is_available()}')
if options.device == "cuda" and torch.cuda.is_available():
    options.device = torch.device("cuda:0")
else:
    options.device = torch.device("cpu")
print(f'Device used: {options.device}')

# Let's first prepare the MNIST dataset,
# run the test_dataset.py file to view some examples and see the dimensions of your tensor.
dataset = Dataset(options)

# define and train the model
model = VanillaAutoEncoder(options)
train_autoencoder(
    model, options, dataset,
    torch.optim.Adam(model.parameters(), lr=options.lr, weight_decay=1e-05)
)

# save the model
save(model, options)

# display some images with its reconstruction
test_autoencoder(model, dataset, options)
generate_using_encoder(model, options)
