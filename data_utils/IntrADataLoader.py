import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import Dataset

def pc_normalize(pc):
    """Normalize the point cloud to zero mean and unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """Sample npoint farthest points from the point cloud."""
    N, D = point.shape
    xyz = point[:, :3]  # Only use XYZ for sampling
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)
    
    point = point[centroids.astype(np.int32)]
    return point

class IntrADataLoader(Dataset):
    def __init__(self, root, num_point,state,num_category, use_uniform_sample, use_normals, split='train', process_data=False):
        """
        Args:
            root: Path to the folder containing the NumPy files.
            num_point: Number of points to sample from each point cloud.
            num_category: Number of categories (not used in NumPy setup, kept for compatibility).
            use_uniform_sample: Whether to use farthest point sampling.
            use_normals: Whether to use normals (set to False if data does not contain normals).
            split: 'train' or 'test'.
            process_data: Whether to process data (optional, not used for NumPy loading).
        """
        self.root = root
        self.npoints = num_point
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.split = split
        self.process_data = process_data
        self.num_category = num_category
        self.state=state
        
        # Load all data and labels
        all_data = np.load(os.path.join(root, 'all_points.npy'))  # Replace with actual path
        all_labels = np.load(os.path.join(root, 'all_labels.npy'))  # Replace with actual path
        if state=='train':
           # Separate the data based on labels (assuming labels are 0 and 1)
           data_label_0 = all_data[all_labels == 0]
           data_label_1 = all_data[all_labels == 1]
           labels_0 = all_labels[all_labels == 0]
           labels_1 = all_labels[all_labels == 1]

           # Split the data and labels for each class (80% train, 20% test)
           train_data_0, test_data_0, train_labels_0, test_labels_0 = train_test_split(data_label_0, labels_0, test_size=0.2, random_state=42)
           train_data_1, test_data_1, train_labels_1, test_labels_1 = train_test_split(data_label_1, labels_1, test_size=0.2, random_state=42)

           # Combine the training data and labels from both classes
           train_data = np.concatenate([train_data_0, train_data_1], axis=0)
           train_labels = np.concatenate([train_labels_0, train_labels_1], axis=0)

           # Combine the test data and labels from both classes
           test_data = np.concatenate([test_data_0, test_data_1], axis=0)
           test_labels = np.concatenate([test_labels_0, test_labels_1], axis=0)

           # Shuffle the training and test sets to ensure random distribution
           train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
       
           test_data, test_labels = shuffle(test_data, test_labels, random_state=42)
        else:
           test_data, test_labels = shuffle(all_data, all_labels, random_state=42)

        # Assign train or test data based on the 'split' argument
        if split == 'train':
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels

        

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        point_set = self.data[index]
        label = self.labels[index]

        # Normalize the point cloud
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # Sample points if needed
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[:self.npoints, :]

        # Use only XYZ if normals are not used
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set.astype(np.float32), int(label)

    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    import torch

    data = IntrADataLoader(
        root='/content/',                  # Path to the folder containing NumPy files
        num_point=1024,                    # Number of points to sample
        num_category=2,                   # Number of categories (optional)
        use_uniform_sample=True,           # Use farthest point sampling
        use_normals=False,                 # Whether the data contains normals
        split='train',                     # Split type ('train' or 'test')
        process_data=False                 # Whether to process data (kept for compatibility)
    )

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
