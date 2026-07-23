import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.data import preprocess_image
from src.intrinsic import get_vision_mle


def extract_embeddings(data_loader, model, device):
    """Extract embeddings for all images in the dataset."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            features = model(images)
            embeddings.append(features.cpu().numpy())
    return np.vstack(embeddings)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet model (use pretrained ResNet18)
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Identity()  # Remove the classification head
    resnet = resnet.to(device)

    # Load EuroSAT_RGB dataset
    data_dir = "data/EuroSAT_RGB"
    dataset = ImageFolder(data_dir, transform=preprocess_image)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(data_loader, resnet, device)

    # Compute MLE for the embeddings
    print("Computing MLE...")
    mle_values = get_vision_mle(embeddings)

    print(f"embeddings shape: {embeddings.shape}")
    print(f"mle_values shape: {mle_values.shape}")
    print(f"mle_values: {mle_values}")

    # Save embeddings and MLE values
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    print(f"Embeddings saved to {output_dir}")


if __name__ == "__main__":
    main()
