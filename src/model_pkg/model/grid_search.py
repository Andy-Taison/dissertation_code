

latent_dims = [2, 4, 8, 16]
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
betas = [0.1, 0.5, 1, 2, 4]
batch_sizes = [32, 64, 128]
optimizer = ["adam", "sgd", "rmsprop"]  # sgd should be tried with momentum, nesterov, weight decay
# momentum
weight_decay = [0, 1e-4, 1e-2]
loss_functions = ["mse", "bce", "smoothl1"]  # bce expects probabilities in range [0,1], otherwise use bcewithlogitsloss as internally applies sigmoid

# loss function type?
# weight decay/ L2 regularisation?

# scheduler parameters (patience, factor)?
# early stopping patience?

# Architecture changes to trial
# hidden units, layers?
# activation functions?
# dropout rates?
# convolution

grid = [
    (latent_dim, lr, beta, batch_size)
    for latent_dim in latent_dims
    for lr in learning_rates
    for beta in betas
    for batch_size in batch_sizes
]