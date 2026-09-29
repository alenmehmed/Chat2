from global_info.global_info import ROOT_DIR
import matplotlib.pyplot as plt
import datetime

def plot_training_and_perplexity(epoch_losses, perplexities, iters_per_epoch):
    x_vals = [i * iters_per_epoch for i in range(len(epoch_losses))]

    get_datetime = datetime.datetime.now()
    timestamp = get_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, epoch_losses, label='Epoch Avg Loss', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.grid(True)

    # Perplexity
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, perplexities, label='Per-Token Perplexity', color='orange', marker='x')
    plt.xlabel('Iteration')
    plt.ylabel('Perplexity')
    plt.title('Perplexity per Epoch')
    plt.grid(True)

    plt.tight_layout()
    plot_path = ROOT_DIR / "training" / "training_plots" / "generated_plots"/ f"loss_perplexity_plot_{timestamp}.png"

    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss + perplexity plot at {plot_path}")
