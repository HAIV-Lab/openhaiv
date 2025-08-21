import argparse
import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms, models

# Argument parsing
parser = argparse.ArgumentParser(description='Image Clustering with NCD Calculation')
parser.add_argument('--n_clusters', type=int, default=95, help='Number of clusters for K-means')
parser.add_argument('--num_classes', type=int, default=95, help='Number of true classes')
parser.add_argument('--id_txt', type=str, default='output/ood/vim_cil_test_0423/exp/0.99id.txt', help='Path to ID text file')
parser.add_argument('--ood_txt', type=str, default='output/ood/vim_cil_test_0423/exp/0.99ood.txt', help='Path to OOD text file')
parser.add_argument('--model_path', type=str, default='', help='Path to model weights')
parser.add_argument('--output_dir', type=str, default='output/ood/vim_cil_test_0423/exp/ncd_data/', help='Output directory')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
parser.add_argument('--random_state', type=int, default=42, help='Random state for K-means')
args = parser.parse_args()


class ImageDataset(Dataset):
    """Dataset class for loading images from file paths"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class FeatureExtractor(nn.Module):
    """Feature extractor using ResNet50 backbone"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        # Remove the final classification layer to get features
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten spatial dimensions
        return x


def batch_extract_features(model, dataloader, device):
    """Extract features from images in batches"""
    features = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            batch = batch.to(device)
            batch_features = model(batch)
            features.append(batch_features.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_ncd_and_remap(true_labels, cluster_labels, n_clusters, num_classes, noise_label=-1):
    """
    Compute Normalized Clustering Distance (NCD) and remap cluster labels
    
    Args:
        true_labels: Ground truth labels (-1 for noise/ID samples)
        cluster_labels: Cluster assignments from K-means
        n_clusters: Number of clusters
        num_classes: Number of true classes
        noise_label: Label value indicating noise/ID samples
    
    Returns:
        ncd: Normalized Clustering Distance value
        remapped_labels: Cluster labels remapped to true label space
    """
    # Filter out noise samples (ID samples with label -1)
    non_noise_mask = true_labels != -1 * np.ones([len(true_labels)])
    filtered_true = np.array(true_labels)[non_noise_mask]
    filtered_cluster = cluster_labels[non_noise_mask]

    if len(filtered_true) == 0:
        raise ValueError("All samples are noise - no OOD samples found")

    # Create mapping from true labels to integer indices
    unique_true = range(0, num_classes)
    true_label_to_int = {label: i for i, label in enumerate(unique_true)}
    true_labels_int = np.array([true_label_to_int[label] for label in filtered_true])

    # Build confusion matrix between clusters and true labels
    confusion_matrix = np.zeros((n_clusters, len(unique_true)))
    for cluster_id in range(n_clusters):
        mask = filtered_cluster == cluster_id
        cluster_true_labels = true_labels_int[mask]
        if len(cluster_true_labels) > 0:
            counts = np.bincount(cluster_true_labels, minlength=len(unique_true))
            confusion_matrix[cluster_id] = counts

    # Use Hungarian algorithm to find optimal cluster-to-true-label assignment
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)  # Negative for maximization

    cluster_to_true = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    print("Cluster to true label mapping:", cluster_to_true)

    # Remap cluster labels to true label space
    remapped_labels = np.array([cluster_to_true.get(c, -1) for c in cluster_labels])

    # Calculate number of correctly matched samples
    matched = confusion_matrix[row_ind, col_ind].sum()
    # Compute NCD: 1 - (number of correct matches / total non-noise samples)
    ncd = 1 - (matched / len(filtered_true))

    return ncd, remapped_labels


def load_images_from_txt(id_txt_path, ood_txt_path, model, transform, batch_size=32):
    """
    Load image paths from text files and extract features
    
    Args:
        id_txt_path: Path to text file containing ID image paths
        ood_txt_path: Path to text file containing OOD image paths
        model: Feature extraction model
        transform: Image transformations
        batch_size: Batch size for feature extraction
    
    Returns:
        image_paths: List of all image paths
        features: Extracted features as numpy array
        true_labels: Ground truth labels (-1 for ID samples)
    """
    image_paths = []
    label_all = []
    
    # Load OOD images (samples with known true labels)
    with open(ood_txt_path, "r") as f:
        for line in f:
            path = line.strip()
            if path and os.path.exists(path):
                image_paths.append(path)
                # Extract label from directory name
                label_all.append(path.split("/")[-2])
    
    # Create label mapping from directory names to integers
    unique_labels = sorted(list(set(label_all)), key=lambda x: x[0].lower())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    true_labels = [label_map[label] for label in label_all]
    
    print(f"Label mapping: {label_map}")
    print(f"Number of OOD samples: {len(image_paths)}")
    
    # Load ID images (treated as noise with label -1)
    with open(id_txt_path, "r") as f:
        for line in f:
            path = line.strip()
            if path and os.path.exists(path):
                image_paths.append(path)
                true_labels.append(-1)  # -1 indicates ID/noise samples
    
    print(f"Total samples (OOD + ID): {len(image_paths)}")
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Extract features using the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    features = batch_extract_features(model, dataloader, device)

    return image_paths, features, true_labels


def main():
    """Main function to execute the clustering pipeline"""
    # Data preprocessing transformations
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and configure the feature extraction model
    model = FeatureExtractor()
    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    # Clean up state dict keys for compatibility
    state_dict = {
        k.replace("network.", "").replace("module.", ""): v for k, v in state_dict.items()
    }
    state_dict = {k.replace("convnet.", "resnet."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # Load images and extract features
    print("Loading images and extracting features...")
    image_paths, X, true_labels = load_images_from_txt(
        args.id_txt, args.ood_txt, model, test_transform, batch_size=args.batch_size
    )

    # Standardize features for better clustering performance
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    print(f"Performing K-means clustering with {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Compute NCD and remap cluster labels to true label space
    print("Computing NCD and remapping labels...")
    ncd_value, remapped_labels = compute_ncd_and_remap(
        true_labels, clusters, args.n_clusters, args.num_classes
    )
    print(f"Normalized Clustering Distance (NCD): {ncd_value:.4f}")

    # Prepare output directory
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Organize images into directories based on remapped cluster labels
    for cluster_id in range(args.n_clusters):
        cluster_mask = clusters == cluster_id
        if np.any(cluster_mask):
            remapped_id = remapped_labels[cluster_mask][0]
            if remapped_id != -1:  # Skip noise clusters
                # Create directory with offset label (94 + remapped_id)
                cluster_dir = os.path.join(args.output_dir, str(94 + remapped_id))
                os.makedirs(cluster_dir, exist_ok=True)

                # Copy all images in this cluster to the directory
                cluster_indices = np.where(clusters == cluster_id)[0]
                for idx in cluster_indices:
                    src_path = image_paths[idx]
                    filename = src_path.split("/")[-1]
                    dest_path = os.path.join(cluster_dir, filename)
                    shutil.copy2(src_path, dest_path)

    # Print summary statistics
    print(f"\nClustering completed! Processed {len(image_paths)} images in total.")
    print(f"Original cluster distribution: {np.bincount(clusters)}")
    print(f"Remapped cluster distribution: {np.bincount(remapped_labels[remapped_labels != -1])}")
    print(f"Normalized Clustering Distance (NCD): {ncd_value:.4f}")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
