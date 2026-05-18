import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import SimpleCNN

SEED = 42
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "mnist_cnn.pth"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            prediction = output.argmax(dim=1)
            correct += prediction.eq(target).sum().item()

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    return average_loss, accuracy


def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    print("Downloading dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
    )

    best_accuracy = 0.0

    print("Starting training loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(data)
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

        train_loss = running_loss / len(train_loader.dataset)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} test_accuracy={test_accuracy * 100:.2f}%"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new best checkpoint to {MODEL_PATH} ({best_accuracy * 100:.2f}%)")
        else:
            print(f"Best checkpoint unchanged ({best_accuracy * 100:.2f}%)")

    print(f"Execution complete. Best test accuracy: {best_accuracy * 100:.2f}%")

if __name__ == "__main__":
    train()
