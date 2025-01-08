import torch
import torch.optim as optim
from .model import VAE, vae_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    # generate random data for testing
    x = torch.randn(64, 784).to(device)

    # train for 1 step
    vae.train()
    optimizer.zero_grad()
    recon_batch, mu, logvar = vae(x)
    loss = vae_loss(recon_batch, x, mu, logvar)
    loss.backward()
    optimizer.step()

    print(f"Training step complete. Loss: {loss.item()}")

    # save model checkpoint
    checkpoint_path = "vae_test_checkpoint.pth"
    torch.save(vae.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # check if checkpoint file can be read
    try:
        loaded_state_dict = torch.load(checkpoint_path)
        vae.load_state_dict(loaded_state_dict)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    if device.type == "cuda":
        print(f"GPU Test: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
    else:
        print("GPU not used.")

    print("Test complete.")
