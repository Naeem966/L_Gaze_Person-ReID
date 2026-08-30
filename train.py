import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import LGazeModel
from losses import CrossEntropyLabelSmooth, TripletLoss, EntropyWeightedContrastiveLoss, TokenDiversityLoss
from dataset import ReIDDataset, ReIDImageDataset, RandomIdentitySampler, build_transforms
from utils import R1_mAP_Evaluator, setup_logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="L-Gaze: Learnable Domain Tokens & Diffusion-Based Feature Stylization for DG-ReID")
    parser.add_argument("--num_classes", type=int, default=64, help="Total cardinality of merged source identities")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs for demonstration")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.008, help="Base learning rate for backbone")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./logs", help="Directory to save logs and checkpoints")
    parser.add_argument("--use_synthetic", action="store_true", default=True, help="Use synthetic dataset for demonstration")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger("L-Gaze", args.save_dir)
    logger.info("Initializing L-Gaze Training Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Dataset & DataLoaders
    if args.use_synthetic:
        logger.info("Using ReIDDataset for multi-epoch execution...")
        train_dataset = ReIDDataset(num_identities=args.num_classes, images_per_id=8)
        val_dataset = ReIDDataset(num_identities=args.num_classes, images_per_id=4)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 2. Model Initialization
    model = LGazeModel(
        num_classes=args.num_classes,
        img_size=(256, 128),
        stride=12,
        num_tokens=50,
        feat_dim=768,
        dfs_hidden_dim=512,
        T=200,
        T_prime=10,
        sigma=0.3
    ).to(device)

    # 3. Loss Functions
    criterion_id = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=0.1).to(device)
    criterion_tri = TripletLoss(margin=0.3).to(device)
    criterion_con = EntropyWeightedContrastiveLoss(temperature=0.07).to(device)
    criterion_div = TokenDiversityLoss(delta=0.0).to(device)

    # Loss weights
    lambda_tri = 1.0
    lambda_con = 0.5
    lambda_div = 0.1

    # 4. Optimizers
    # Optimizer for backbone + BNNeck + Classifier (SGD)
    backbone_params = list(model.backbone.parameters()) + list(model.head.parameters())
    optimizer_backbone = optim.SGD(
        backbone_params, lr=args.lr, momentum=0.9, weight_decay=1e-4
    )
    
    # Optimizer for DFS module (Adam)
    optimizer_dfs = optim.Adam(
        model.dfs.parameters(), lr=1e-3, weight_decay=1e-4
    )

    # Optimizer for LDT bank (Adam)
    optimizer_ldt = optim.Adam(
        model.ldt_bank.parameters(), lr=1e-2, weight_decay=0.0
    )

    evaluator = R1_mAP_Evaluator(num_query=len(val_dataset) // 2)

    logger.info("Starting Multi-Phase Training (Algorithm 1)...")
    logger.info("Phase 1: Epochs 1 - 5 (Backbone Warm-up)")
    logger.info("Phase 2a: Epochs 6 - 10 (DFS & LDT Reconstruction Warm-up)")
    logger.info("Phase 3: Epochs 11 - Total (Joint Optimization under L_total)")

    phase1_epochs = max(1, int(args.epochs * 0.2))
    phase2_epochs = max(2, int(args.epochs * 0.35))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        loss_id_epoch = 0.0
        loss_tri_epoch = 0.0
        loss_con_epoch = 0.0
        loss_div_epoch = 0.0

        if epoch <= phase1_epochs:
            # Phase 1: Backbone Warm-up
            current_phase = "Phase 1 (Backbone Warmup)"
            # Linear LR warmup
            lr_factor = float(epoch) / float(phase1_epochs)
            for param_group in optimizer_backbone.param_groups:
                param_group['lr'] = args.lr * lr_factor

            for imgs, pids, camids, _ in train_loader:
                imgs, pids = imgs.to(device), pids.to(device)
                
                optimizer_backbone.zero_grad()
                
                z_cls = model.backbone(imgs)
                f_BN, logits = model.head(z_cls)

                loss_id = criterion_id(logits, pids)
                loss_tri = criterion_tri(f_BN, pids)
                loss_total = loss_id + lambda_tri * loss_tri

                loss_total.backward()
                optimizer_backbone.step()

                total_loss_epoch += loss_total.item()
                loss_id_epoch += loss_id.item()
                loss_tri_epoch += loss_tri.item()

        elif epoch <= phase2_epochs:
            # Phase 2a: DFS & LDT Warm-up via Reconstruction
            current_phase = "Phase 2a (DFS Warmup)"
            # Reduce backbone LR
            for param_group in optimizer_backbone.param_groups:
                param_group['lr'] = args.lr * 0.1

            for imgs, pids, camids, _ in train_loader:
                imgs = imgs.to(device)
                
                # Freeze backbone, train DFS on feature reconstruction
                with torch.no_grad():
                    f_s = model.backbone(imgs)
                    # Simulated feature augmentation for reconstruction
                    f_aug = f_s + torch.randn_like(f_s) * 0.05

                optimizer_dfs.zero_grad()
                optimizer_ldt.zero_grad()

                # Reconstruction f_n = f_aug + Delta f(f_aug, 0)
                hat_f_n = model.forward_dfs_warmup(f_aug)
                loss_rec = torch.mean((f_s - hat_f_n) ** 2)
                
                # LDT diversity loss
                token_bank = model.ldt_bank.get_bank()
                loss_div = criterion_div(token_bank)

                loss_dfs = loss_rec + lambda_div * loss_div
                loss_dfs.backward()
                
                optimizer_dfs.step()
                optimizer_ldt.step()

                total_loss_epoch += loss_rec.item()
                loss_div_epoch += loss_div.item()

        else:
            # Phase 3: Joint Fine-tuning under L_total
            current_phase = "Phase 3 (Joint Optimization)"
            # Cosine annealing schedule
            progress = float(epoch - phase2_epochs) / float(args.epochs - phase2_epochs)
            lr_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            for param_group in optimizer_backbone.param_groups:
                param_group['lr'] = args.lr * lr_factor

            for imgs, pids, camids, _ in train_loader:
                imgs, pids = imgs.to(device), pids.to(device)
                
                optimizer_backbone.zero_grad()
                optimizer_dfs.zero_grad()
                optimizer_ldt.zero_grad()

                # Full training pass
                out = model.forward_train(imgs)
                z_cls = out['z_cls']
                f_BN = out['f_BN']
                logits = out['logits']
                f_n = out['f_n']
                token_bank = out['token_bank']

                loss_id = criterion_id(logits, pids)
                loss_tri = criterion_tri(f_BN, pids)
                loss_con = criterion_con(z_cls, f_n, pids, model.head.classifier, model.head.bottleneck, args.num_classes)
                loss_div = criterion_div(token_bank)

                loss_total = loss_id + lambda_tri * loss_tri + lambda_con * loss_con + lambda_div * loss_div

                loss_total.backward()

                optimizer_backbone.step()
                optimizer_dfs.step()
                optimizer_ldt.step()

                total_loss_epoch += loss_total.item()
                loss_id_epoch += loss_id.item()
                loss_tri_epoch += loss_tri.item()
                loss_con_epoch += loss_con.item()
                loss_div_epoch += loss_div.item()

        num_batches = len(train_loader)
        avg_loss = total_loss_epoch / num_batches
        avg_id = loss_id_epoch / num_batches
        avg_tri = loss_tri_epoch / num_batches
        avg_con = loss_con_epoch / num_batches
        avg_div = loss_div_epoch / num_batches

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] {current_phase} - "
            f"Total Loss: {avg_loss:.4f} | ID: {avg_id:.4f} | Tri: {avg_tri:.4f} | "
            f"Con: {avg_con:.4f} | Div: {avg_div:.4f}"
        )

        # Evaluate target-domain accuracy every 5 epochs or on final epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            logger.info(f"--- Target Domain Evaluation at Epoch {epoch} ---")
            val_res = evaluator.evaluate(model, val_loader, device)
            logger.info(
                f"[Evaluation Results] Target mAP: {val_res['mAP']:.2f}% | "
                f"Rank-1: {val_res['Rank-1']:.2f}% | Rank-5: {val_res['Rank-5']:.2f}% | Rank-10: {val_res['Rank-10']:.2f}%"
            )

    logger.info("L-Gaze Training Complete!")
    # Save checkpoint
    ckpt_path = os.path.join(args.save_dir, "lgaze_final.pth")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    main()
