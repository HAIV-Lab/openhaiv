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
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_ncd_and_remap(true_labels, cluster_labels, n_clusters, noise_label=-1):
    non_noise_mask = (true_labels != -1 * np.ones([len(true_labels)]))
    filtered_true = np.array(true_labels)[non_noise_mask]
    filtered_cluster = cluster_labels[non_noise_mask]

    if len(filtered_true) == 0:
        raise ValueError("all sample noise")

    unique_true = range(0,95)
    true_label_to_int = {label: i for i, label in enumerate(unique_true)}
    true_labels_int = np.array([true_label_to_int[label] for label in filtered_true])

    confusion_matrix = np.zeros((n_clusters, len(unique_true)))
    for cluster_id in range(n_clusters):
        mask = (filtered_cluster == cluster_id)
        cluster_true_labels = true_labels_int[mask]
        if len(cluster_true_labels) > 0:
            counts = np.bincount(cluster_true_labels, minlength=len(unique_true))
            confusion_matrix[cluster_id] = counts

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    cluster_to_true = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    print(cluster_to_true)

    remapped_labels = np.array([cluster_to_true[c] for c in cluster_labels])

    matched = confusion_matrix[row_ind, col_ind].sum()

    ncd = 1 - (matched / len(filtered_true))

    return ncd, remapped_labels


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def batch_extract_features(model, dataloader, device):
    features = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch_features = model(batch)
            features.append(batch_features.cpu().numpy())
    return np.concatenate(features, axis=0)


def load_images_from_txt(id_txt_path, ood_txt_path, model, transform, batch_size=32):
    image_paths = []
    label_all = []
    with open(ood_txt_path, 'r') as f:
        for line in f:
            path = line.strip()
            if path and os.path.exists(path):
                image_paths.append(path)
                label_all.append(path.split('/')[-2])
    unique_labels = sorted(list(set(label_all)), key=lambda x: x[0].lower())

    label_map = {label: i for i, label in enumerate(unique_labels)}
    true_labels = [label_map[label] for label in label_all]
    print(label_map)
    with open(id_txt_path, 'r') as f:
        for line in f:
            path = line.strip()
            if path and os.path.exists(path):
                image_paths.append(path)
                true_labels.append(-1)

    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    features = batch_extract_features(model, dataloader, device)

    return image_paths, features, true_labels


test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = FeatureExtractor()
weight_path = "/data/zqh/openhaiv/output/supervised/inc_BM200_finetune_1/exp/task_0.pth"
checkpoint = torch.load(weight_path)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
state_dict = {k.replace('network.', '').replace('module.', ''): v for k, v in state_dict.items()}
state_dict = {k.replace('convnet.', 'resnet.'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

id_txt_file = "/data/zqh/xz/openhaiv-ood_xz/output/ood/vim_cil_test_0423/exp/0.99id.txt"
ood_txt_file = "/data/zqh/xz/openhaiv-ood_xz/output/ood/vim_cil_test_0423/exp/0.99ood.txt"
image_paths, X, true_labels = load_images_from_txt(id_txt_file, ood_txt_file, model, test_transform, batch_size=64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = 95
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

ncd_value, remapped_labels = compute_ncd_and_remap(true_labels, clusters, n_clusters)
print(f"Normalized Clustering Distance (NCD): {ncd_value:.4f}")

output_dir = "/data/zqh/xz/openhaiv-ood_xz/output/ood/vim_cil_test_0423/exp/ncd_data/"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

for cluster_id in range(n_clusters):
    remapped_id = remapped_labels[clusters == cluster_id][0]
    if remapped_id == -1:
        pass
    cluster_dir = os.path.join(output_dir, str(94+remapped_id))
    os.makedirs(cluster_dir, exist_ok=True)

    cluster_indices = np.where(clusters == cluster_id)[0]
    for idx in cluster_indices:
        src_path = image_paths[idx]
        true_label = true_labels[idx]
        filename = src_path.split('/')[-1]
        dest_path = os.path.join(cluster_dir, filename)

        shutil.copy2(src_path, dest_path)

print(f"Clustering completed! Processed {len(image_paths)} images in total.")
print(f"Original cluster distribution: {np.bincount(clusters)}")
print(f"Remapped cluster distribution: {np.bincount(remapped_labels[remapped_labels != -1])}")  # Ignore unmatched labels (-1)
print(f"Normalized Clustering Distance (NCD): {ncd_value:.4f}")
print(f"Results saved to: {os.path.abspath(output_dir)}")