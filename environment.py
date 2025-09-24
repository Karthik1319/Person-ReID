import torch

class ReIDEnvironment:
    def __init__(self, reid_model, data_loader, device):
        self.reid_model = reid_model
        self.data_loader = data_loader
        self.device = device
        self.data_iterator = iter(self.data_loader)

    def get_state(self):
        anchor, pos, neg = next(self.data_iterator)
        return anchor.to(self.device), pos.to(self.device), neg.to(self.device)

    def step(self, state_triplet, action_mask):
        anchor, pos, neg = state_triplet
        with torch.no_grad():
            anchor_feats = self.reid_model.extract_features(anchor) * action_mask
            pos_feats = self.reid_model.extract_features(pos)
            neg_feats = self.reid_model.extract_features(neg)
            dist_pos = torch.pairwise_distance(anchor_feats, pos_feats, p=2)
            dist_neg = torch.pairwise_distance(anchor_feats, neg_feats, p=2)
            reward = -(torch.relu(dist_pos - dist_neg).mean().item())
        return reward

    def reset(self):
        self.data_iterator = iter(self.data_loader)