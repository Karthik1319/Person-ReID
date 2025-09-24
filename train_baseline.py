import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import numpy as np

# --- Local Imports from the 'reid' package ---
from model import FeatureExtractor, ReIDModel
from data_loader import Market1501
from distances import mahalanobis_dist_from_features

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This class acts as a bridge between the Market1501 loader and a standard DataLoader
class MarketClassificationDataset(Dataset):
    def __init__(self, data_path):
        self.market_dataset = Market1501(data_path=data_path, mode='train')
        self.paths = self.market_dataset.paths
        self.labels = self.market_dataset.labels
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        img = self.market_dataset._process(path)
        return img, label

# --- NEW: Helper function for Online Triplet Mining ---
def select_hard_triplets(features, labels):
    """
    Selects the hardest positive and hardest negative for each anchor in the batch.
    
    Args:
        features (torch.Tensor): The feature vectors for the batch (BatchSize x FeatureDim).
        labels (torch.Tensor): The labels for the batch (BatchSize).

    Returns:
        Tuple of Tensors: (anchor_features, positive_features, negative_features)
    """
    # Calculate pairwise distance matrix
    pairwise_dist = mahalanobis_dist_from_features(features)

    anchors, positives, negatives = [], [], []
    for i in range(len(labels)):
        anchor_label = labels[i]
        
        # Find indices of all positives and negatives for the current anchor
        pos_mask = (labels == anchor_label)
        pos_mask[i] = False # Exclude the anchor itself
        neg_mask = (labels != anchor_label)

        # If no other positives or no negatives in batch, skip this anchor
        if not torch.any(pos_mask) or not torch.any(neg_mask):
            continue

        # Select the hardest positive (one with the largest distance)
        hardest_positive_dist = torch.max(pairwise_dist[i][pos_mask])
        hardest_positive_idx = torch.where((pairwise_dist[i] == hardest_positive_dist) & pos_mask)[0][0]

        # Select the hardest negative (one with the smallest distance)
        hardest_negative_dist = torch.min(pairwise_dist[i][neg_mask])
        hardest_negative_idx = torch.where((pairwise_dist[i] == hardest_negative_dist) & neg_mask)[0][0]

        anchors.append(features[i])
        positives.append(features[hardest_positive_idx])
        negatives.append(features[hardest_negative_idx])

    if not anchors:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def main(args):
    print(f"Using device: {DEVICE}")

    # --- 1. Setup Data Loader ---
    try:
        train_dataset = MarketClassificationDataset(data_path=args.data_path)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=2,
            drop_last=True # Important for triplet mining
        )
        print("Data loader ready.")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n!!! DATASET ERROR: {e} !!!")
        return

    # --- 2. Setup Model ---
    feature_extractor = FeatureExtractor(pretrained=True)
    num_classes = train_dataset.market_dataset.num_classes
    model = ReIDModel(feature_extractor, num_classes=num_classes).to(DEVICE)
    model.train()

    # --- 3. Setup Loss and Optimizer ---
    # --- NEW: Using a HYBRID loss ---
    criterion_ce = nn.CrossEntropyLoss()
    criterion_triplet = nn.TripletMarginLoss(margin=0.3, p=2)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # --- 4. Supervised Training Loop ---
    print("Starting supervised fine-tuning with HYBRID loss...")
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            features, class_scores = model(images)
            
            # --- NEW: Calculate both losses ---
            loss_ce = criterion_ce(class_scores, labels)
            
            # Select triplets for the triplet loss
            anchors, positives, negatives = select_hard_triplets(features, labels)
            
            if anchors is not None:
                loss_triplet = criterion_triplet(anchors, positives, negatives)
                # Combine the losses
                loss = loss_ce + loss_triplet
            else:
                # Fallback to only CE loss if no valid triplets were found in the batch
                loss_triplet = torch.tensor(0.0)
                loss = loss_ce

            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({
                'Total Loss': f'{loss.item():.3f}',
                'CE Loss': f'{loss_ce.item():.3f}',
                'Triplet Loss': f'{loss_triplet.item():.3f}'
            })

    print("Finished fine-tuning.")
    
    # --- 5. Save the Fine-Tuned Feature Extractor Weights ---
    torch.save(model.feature_extractor.state_dict(), args.save_path)
    print(f"Saved fine-tuned feature extractor weights to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a feature extractor for Person Re-ID.")
    parser.add_argument('--data_path', type=str, default='./Market-1501-v15.09.15/', help='Path to Market-1501 dataset')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train. (Increased for better convergence)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--save_path', type=str, default='extractor_finetuned.pth', help='Path to save the fine-tuned weights.')
    args = parser.parse_args()
    main(args)
