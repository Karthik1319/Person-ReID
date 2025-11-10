import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from tqdm import tqdm

# --- Local Imports from the 'reid' package ---
from model import FeatureExtractor, PolicyNetwork, ReIDModel
from environment import ReIDEnvironment
from agent import PolicyGradientAgent
from data_loader import Market1501

# --- Configuration ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# --- Helper Dataset Class to Create Triplets ---
class MarketTripletDataset(Dataset):
    def __init__(self, market_dataset, num_episodes):
        self.market_dataset = market_dataset
        self.num_episodes = num_episodes
        # We need the raw PIDs for finding correct positive/negative pairs
        self.pids_raw = self.market_dataset.pids_raw
        self.paths = self.market_dataset.paths

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, index):
        while True:
            # Sample a query image
            q_idx = np.random.randint(0, len(self.paths))
            q_path, q_pid = self.paths[q_idx], self.pids_raw[q_idx]

            # Find a positive and negative sample
            pids_array = np.array(self.pids_raw)
            pos_indices = np.where(pids_array == q_pid)[0]
            neg_indices = np.where(pids_array != q_pid)[0]

            # Ensure we don't pick the query image itself as the positive
            pos_indices = pos_indices[pos_indices != q_idx]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = np.random.choice(pos_indices)
                neg_idx = np.random.choice(neg_indices)
                
                anchor_img = self.market_dataset._process(self.paths[q_idx])
                pos_img = self.market_dataset._process(self.paths[pos_idx])
                neg_img = self.market_dataset._process(self.paths[neg_idx])

                return anchor_img, pos_img, neg_img

def main(args):
    print(f"Using device: {DEVICE}")

    # --- 1. Setup Models ---
    feature_extractor = FeatureExtractor(pretrained=True).to(DEVICE)
    try:
        feature_extractor.load_state_dict(torch.load('extractor_finetuned.pth', map_location=DEVICE))
        print("Successfully loaded fine-tuned feature extractor weights.")
    except FileNotFoundError:
        print("Warning: Fine-tuned weights not found. Using default ImageNet weights for feature extractor.")

    feature_extractor.eval() # Freeze the extractor
    policy_network = PolicyNetwork().to(DEVICE)
    
    reid_model = ReIDModel(feature_extractor, num_classes=args.num_classes).to(DEVICE)

    # --- 2. Setup Data Loader ---
    print(f"Setting up Market-1501 data loader from path: {args.data_path}")
    try:
        market_dataset = Market1501(data_path=args.data_path, mode='train')
        triplet_dataset = MarketTripletDataset(market_dataset, num_episodes=args.num_episodes)
        train_loader = DataLoader(triplet_dataset, batch_size=args.batch_size, shuffle=False)
        print("Data loader ready.")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n!!! DATASET ERROR: {e} !!!")
        print(f"Could not find training data in path: '{args.data_path}'")
        return

    # --- 3. Setup RL Agent and Environment ---
    agent = PolicyGradientAgent(policy_network, feature_extractor, learning_rate=args.learning_rate)
    environment = ReIDEnvironment(reid_model, train_loader, DEVICE)

    # --- 4. Training Loop ---
    print("Starting RL training...")
    total_rewards = []
    progress_bar = tqdm(range(args.num_episodes), desc="Training")
    for episode in progress_bar:
        try:
            state_triplet = environment.get_state()
            action = agent.select_action(state_triplet[0]) # Action based on anchor
            reward = environment.step(state_triplet, action)
            agent.rewards.append(reward)
            agent.update_policy()
            total_rewards.append(reward)
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                progress_bar.set_postfix({'Avg Reward': f'{avg_reward:.4f}'})
        except StopIteration:
            print("Restarting data loader.")
            environment.reset()

    print("Training finished.")
    torch.save(policy_network.state_dict(), "policy_network_reid.pth")
    print("Saved trained policy network to policy_network_reid.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Policy Gradient Agent for Person Re-ID.")
    parser.add_argument('--data_path', type=str, default='./Market-1501-v15.09.15/', help='Path to Market-1501 dataset')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Total training episodes (batches).')
    parser.add_argument('--batch_size', type=int, default=16, help='Triplets per batch.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for policy network.')
    parser.add_argument('--num_classes', type=int, default=751, help='Number of classes in the training set.')
    args = parser.parse_args()
    main(args)
