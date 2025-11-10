import torch
import numpy as np
import argparse
from tqdm import tqdm

# --- Local Imports from the 'reid' package ---
from model import FeatureExtractor, PolicyNetwork, ReIDModel
from data_loader import Market1501
from utils import evaluate_ranked_list
from distances import mahalanobis_dist_from_vectors

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def evaluate_reid(model, policy_network, dataloader, use_rl=False):
    model.eval()
    if policy_network: policy_network.eval()

    print("Extracting features...")
    gallery_imgs, gallery_labels, gallery_cams = dataloader.get_full_gallery()
    gallery_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(gallery_imgs), 64), desc="Gallery Features"):
            batch = torch.stack(gallery_imgs[i:i+64]).to(DEVICE)
            gallery_feats.append(model.extract_features(batch).cpu())
    gallery_feats = torch.cat(gallery_feats, dim=0)

    query_imgs, query_labels, query_cams = dataloader.get_full_query()
    query_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(query_imgs), 64), desc="Query Features"):
            batch = torch.stack(query_imgs[i:i+64]).to(DEVICE)
            query_feats.append(model.extract_features(batch).cpu())
    query_feats = torch.cat(query_feats, dim=0)
    
    all_aps, cmc_curve = [], np.zeros(len(gallery_labels))
    for i in tqdm(range(len(query_feats)), desc="Evaluating Queries"):
        q_feat, q_label, q_cam = query_feats[i], query_labels[i], query_cams[i]
        q_feat_final = q_feat.to(DEVICE)

        if use_rl and policy_network:
            with torch.no_grad():
                feature_map = model.feature_extractor(query_imgs[i].unsqueeze(0).to(DEVICE))
                mean, _ = policy_network(feature_map)
                attention_mask = torch.sigmoid(mean).squeeze(0).detach()
            q_feat_final *= attention_mask

        distances = mahalanobis_dist_from_vectors(q_feat_final.cpu(), gallery_feats)
        ranking = torch.argsort(distances).numpy()
        ap, cmc = evaluate_ranked_list(ranking, np.array(gallery_labels), np.array(gallery_cams), q_label, q_cam)
        all_aps.append(ap)
        cmc_curve += cmc

    mAP = np.mean(all_aps) * 100.0
    cmc_scores = (cmc_curve / len(query_feats)) * 100.0
    return mAP, cmc_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Re-ID model with an optional RL policy.")
    parser.add_argument('--data_path', type=str, default='./Market-1501-v15.09.15/', help='Path to Market-1501 dataset')
    parser.add_argument('--policy_path', type=str, default='policy_network_reid.pth', help='Path to trained policy network.')
    # --- ADDED: num_classes argument for consistency ---
    parser.add_argument('--num_classes', type=int, default=751, help='Number of classes in the training set.')
    args = parser.parse_args()

    feature_extractor = FeatureExtractor(pretrained=True).to(DEVICE)
    # --- FIX: Pass the num_classes argument to the ReIDModel constructor ---
    reid_model = ReIDModel(feature_extractor, num_classes=args.num_classes).to(DEVICE)
    
    try:
        # Load the fine-tuned weights into the feature extractor part of the model
        reid_model.feature_extractor.load_state_dict(torch.load('extractor_finetuned.pth', map_location=DEVICE))
        print("Successfully loaded fine-tuned feature extractor weights.")
    except FileNotFoundError:
        print("Warning: Fine-tuned weights not found. Using default ImageNet weights for baseline.")
    
    try:
        policy_network = PolicyNetwork().to(DEVICE)
        policy_network.load_state_dict(torch.load(args.policy_path, map_location=DEVICE))
        print(f"Successfully loaded policy network from '{args.policy_path}'")
        policy_loaded = True
    except FileNotFoundError:
        print(f"Warning: Policy network not found. Running baseline evaluation only.")
        policy_network, policy_loaded = None, False

    try:
        test_loader = Market1501(data_path=args.data_path, mode='test')
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n!!! TEST DATASET ERROR: {e} !!!")
        return

    print("\n" + "="*50 + "\nRunning BASELINE evaluation (no RL)...\n" + "="*50)
    baseline_map, baseline_cmc = evaluate_reid(reid_model, None, test_loader, use_rl=False)
    print("\n--- Baseline Results ---")
    print(f"mAP: {baseline_map:.2f}%")
    print(f"Rank-1: {baseline_cmc[0]:.2f}% | Rank-5: {baseline_cmc[4]:.2f}% | Rank-10: {baseline_cmc[9]:.2f}%")

    if policy_loaded:
        print("\n" + "="*50 + "\nRunning evaluation WITH RL Policy...\n" + "="*50)
        rl_map, rl_cmc = evaluate_reid(reid_model, policy_network, test_loader, use_rl=True)
        print("\n--- RL Policy Results ---")
        print(f"mAP: {rl_map:.2f}%")
        # --- FIX: Corrected the typo in the f-string ---
        print(f"Rank-1: {rl_cmc[0]:.2f}% | Rank-5: {rl_cmc[4]:.2f}% | Rank-10: {rl_cmc[9]:.2f}%")
        print("\n" + "="*50 + "\nPerformance Improvement\n" + "="*50)
        print(f"mAP Improvement:       {rl_map - baseline_map:+.2f}%")
        print(f"Rank-1 Improvement:    {rl_cmc[0] - baseline_cmc[0]:+.2f}%")

if __name__ == "__main__":
    main()
