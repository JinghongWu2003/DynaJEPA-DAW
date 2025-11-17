import argparse
import torch
from tqdm import tqdm

from src.data import get_labeled_loaders
from src.models import JEPAModel
from src.utils.checkpoint import load_checkpoint
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="k-NN evaluation for pretrained JEPA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_features(model: JEPAModel, loader, device: str):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc="Extract", leave=False):
            images = images.to(device)
            f = model.online_encoder(images)
            feats.append(f.cpu())
            labels.append(targets)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def knn_classifier(train_feats, train_labels, test_feats, k: int = 200):
    train_norm = torch.nn.functional.normalize(train_feats, dim=1)
    test_norm = torch.nn.functional.normalize(test_feats, dim=1)
    sims = torch.matmul(test_norm, train_norm.t())
    topk = sims.topk(k=k, dim=1).indices
    topk_labels = train_labels[topk]
    preds = []
    for labels in topk_labels:
        vals, counts = labels.unique(return_counts=True)
        preds.append(vals[counts.argmax()])
    return torch.stack(preds)


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("knn_eval")

    train_loader, test_loader = get_labeled_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size
    )

    device = args.device
    model = JEPAModel()
    ckpt = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    train_feats, train_labels = extract_features(model, train_loader, device)
    test_feats, test_labels = extract_features(model, test_loader, device)

    preds = knn_classifier(train_feats, train_labels, test_feats, k=args.k)
    acc = 100.0 * (preds == test_labels).sum().item() / len(test_labels)
    logger.info(f"k-NN accuracy (k={args.k}): {acc:.2f}%")


if __name__ == "__main__":
    main()
