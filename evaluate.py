import os
import argparse
import torch
from torch.utils.data import DataLoader
from models import LGazeModel
from dataset import ReIDDataset
from utils import R1_mAP_Evaluator, setup_logger

def main():
    parser = argparse.ArgumentParser(description="L-Gaze Inference & Evaluation")
    parser.add_argument("--weights", type=str, default="", help="Path to checkpoint weights")
    parser.add_argument("--num_classes", type=int, default=64, help="Number of classes used during training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    logger = setup_logger("L-Gaze-Eval", None)
    logger.info("Initializing L-Gaze Evaluator...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation Device: {device}")

    # Initialize model
    model = LGazeModel(
        num_classes=args.num_classes,
        img_size=(256, 128),
        stride=12,
        num_tokens=50,
        feat_dim=768
    ).to(device)

    if args.weights and os.path.exists(args.weights):
        logger.info(f"Loading checkpoint weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        logger.info("No checkpoint weights provided or file not found. Running zero-shot evaluation...")

    val_dataset = SyntheticReIDDataset(num_identities=args.num_classes, images_per_id=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    evaluator = R1_mAP_Evaluator(num_query=len(val_dataset) // 2)

    logger.info("Running zero-target-domain inference (DFS & LDT modules disabled)...")
    results = evaluator.evaluate(model, val_loader, device)

    logger.info("=== Final Evaluation Metrics ===")
    logger.info(f"Mean Average Precision (mAP): {results['mAP']:.2f}%")
    logger.info(f"Rank-1 Accuracy:  {results['Rank-1']:.2f}%")
    logger.info(f"Rank-5 Accuracy:  {results['Rank-5']:.2f}%")
    logger.info(f"Rank-10 Accuracy: {results['Rank-10']:.2f}%")

if __name__ == "__main__":
    main()
