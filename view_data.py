import matplotlib.pyplot as plt
from torchvision import datasets

def view_local_data():
    # 1. Point PyTorch to your local ./data folder. 
    # We do NOT use transforms here because we want to see the raw image, not the tensor math.
    print("Loading local MNIST data...")
    train_dataset = datasets.MNIST('./data', train=True, download=False)

    # 2. Create a grid to show 6 images
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))

    # 3. Loop through the first 6 items in the dataset
    for i in range(6):
        # PyTorch datasets return a tuple: (The Image, The Label)
        image, label = train_dataset[i]
        
        # Plot it
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    print("Opening viewer...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_local_data()