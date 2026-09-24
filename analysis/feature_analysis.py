import argparse
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_encoder(weights_path, device):

    model = timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=0,
        in_chans=1
    )

    checkpoint = torch.load(
        weights_path,
        map_location="cpu"
    )

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    cleaned = {}

    for key, value in checkpoint.items():

        if key.startswith("momentum_backbone."):
            key = key.replace("momentum_backbone.", "", 1)

        elif key.startswith("backbone."):
            key = key.replace("backbone.", "", 1)

        cleaned[key] = value

    model.load_state_dict(cleaned, strict=False)

    model.eval()
    model.to(device)

    return model


def extract_features(model, image_paths, device):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

    features = []
    names = []

    with torch.no_grad():

        for image_path in image_paths:

            image = Image.open(image_path).convert("L")
            image = transform(image)
            image = image.unsqueeze(0).to(device)

            feature = model(image)

            feature = feature.flatten(1)

            features.append(feature.cpu())
            names.append(image_path.name)

    return torch.cat(features).numpy(), names


def create_pca_plot(features, names, output_path):

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        alpha=0.8
    )

    for i, name in enumerate(names):

        plt.annotate(
            name[:15],
            (reduced[i, 0], reduced[i, 1]),
            fontsize=7
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(
        "Self-Supervised Medical Image Representation Analysis"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    explained = pca.explained_variance_ratio_

    print("\nPCA analysis complete.")

    print(
        f"Variance explained: "
        f"{explained[0] * 100:.2f}% + "
        f"{explained[1] * 100:.2f}%"
    )

    print(
        f"Saved visualization to: {output_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing medical images"
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to SSL-MedSeg pretrained weights"
    )

    parser.add_argument(
        "--output",
        default="analysis_results"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    weights_path = Path(args.weights)
    output_dir = Path(args.output)

    output_dir.mkdir(
        exist_ok=True
    )

    image_paths = []

    for extension in [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp"
    ]:

        image_paths.extend(
            input_dir.glob(extension)
        )

    if len(image_paths) < 3:

        raise ValueError(
            "Please provide at least 3 medical images."
        )

    if not weights_path.exists():

        raise FileNotFoundError(
            f"Weights not found: {weights_path}"
        )

    print(
        f"Found {len(image_paths)} images."
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    print(
        "Loading SSL-MedSeg pretrained encoder..."
    )

    model = load_encoder(
        weights_path,
        device
    )

    features, names = extract_features(
        model,
        image_paths,
        device
    )

    torch.save(
        torch.tensor(features),
        output_dir / "image_embeddings.pt"
    )

    create_pca_plot(
        features,
        names,
        output_dir / "representation_pca.png"
    )

    print(
        "\nFEATURE ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
    