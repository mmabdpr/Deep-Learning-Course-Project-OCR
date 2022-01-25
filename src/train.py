from distutils.command import check
import torch
from torch import nn
from tqdm import tqdm
from src.crnn import CRNN48
from src.dataset import IdDataset

TOTAL_BATCHES = 100000
BATCH_SIZE = 512
# CHECKPOINT_PATH = "drive/MyDrive/ocr/Checkpoints"
CHECKPOINT_PATH = "data/checkpoints"
OUTPUT_HEIGHT = 48
LOAD_FROM_CHECKPOINT = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_batch(model: torch.nn.Module, criterion, optimizer: torch.optim.Optimizer, batch, target, target_lengths):
    preds: torch.Tensor = model.forward(batch)
    preds_size = torch.IntTensor([preds.size(0)] * BATCH_SIZE)
    cost: torch.Tensor = criterion(preds, target, preds_size, target_lengths)
    model.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == '__main__':
    model = CRNN48().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    i = 0
    history = []
    
    if LOAD_FROM_CHECKPOINT:
        checkpoint = torch.load(LOAD_FROM_CHECKPOINT)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['Optimizer'])
        i = checkpoint['Iteration']
        history = checkpoint['History']

    dataset = IdDataset(batch_size=BATCH_SIZE, output_height=OUTPUT_HEIGHT)
    datasetIterator = iter(dataset)
    pbar = tqdm(range(i, TOTAL_BATCHES))
    for iter in pbar:
        x, y = next(datasetIterator)
        if i % 500 == 1:
            torch.save({
                "model": model.state_dict(),
                "Iteration": iter,
                "Loss": cost,
                "Optimizer": optimizer.state_dict(),
                "History": history
            }, f"{CHECKPOINT_PATH}/{i}.torch")

        x = x.to(device)
        y = y.to(device)

        y_lengths = torch.full(
            size=(y.shape[0],), fill_value=y.shape[1], dtype=torch.int32).to(device)

        cost = train_batch(model, criterion, optimizer, x, y, y_lengths)
        history.append(cost.cpu().detach().numpy())
        pbar.set_description(f"loss: {history[-1]:.2f}")

    
    torch.save({
                "model": model.state_dict(),
                "Iteration": i,
                "Loss": cost,
                "Optimizer": optimizer.state_dict(),
                "History": history
            }, f"{CHECKPOINT_PATH}/{i}.torch")
