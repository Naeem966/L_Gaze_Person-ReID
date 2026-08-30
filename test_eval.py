import torch
from dataset import ReIDDataset
from models import LGazeModel
from utils import R1_mAP_Evaluator

def test_eval():
    model = LGazeModel(num_classes=16, img_size=(256, 128), stride=16, feat_dim=768)
    model.eval()
    
    val_dataset = SyntheticReIDDataset(num_identities=16, images_per_id=6, num_cameras=4)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    evaluator = R1_mAP_Evaluator(num_query=0)
    results = evaluator.evaluate(model, loader, torch.device("cpu"))
    print(f"Fixed Evaluation Output: mAP = {results['mAP']:.2f}%, Rank-1 = {results['Rank-1']:.2f}%")

if __name__ == "__main__":
    test_eval()
