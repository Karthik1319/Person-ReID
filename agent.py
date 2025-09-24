import torch
import torch.optim as optim
from torch.distributions import Normal

class PolicyGradientAgent:
    def __init__(self, policy_network, feature_extractor, learning_rate=1e-4, gamma=0.99):
        self.policy_network = policy_network
        self.feature_extractor = feature_extractor
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state_image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature_map = self.feature_extractor(state_image) 
        
        mean, std = self.policy_network(feature_map)
        action_dist = Normal(mean, std)
        action = action_dist.sample()
        
        self.log_probs.append(action_dist.log_prob(action).sum(dim=-1))
        return torch.sigmoid(action)

    def update_policy(self):
        if not self.rewards: return

        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
            
        log_probs = torch.cat(self.log_probs)
        rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=log_probs.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9) if len(rewards) > 1 else rewards
        
        policy_loss = (-log_probs * rewards).sum()
            
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.clear_memory()

    def clear_memory(self):
        self.log_probs.clear()
        self.rewards.clear()