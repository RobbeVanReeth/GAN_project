class Options:
    def __init__(self):
        self.dataset = "MNIST"
        self.batch_size_test = 1000
        self.batch_size_train = 64
        self.random_seed = 1
        self.num_epochs = 3
        self.device = "cpu"
        self.save_path = "models/"
        self.load_path = "models/"
        self.model_name = "ae_latent_dim_2.pth"
        self.encoded_space_dim = 2
        self.lr = 0.001
        self.lamda = 0.00005
