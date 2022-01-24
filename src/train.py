from ast import mod
import torch
from torch import nn
from tqdm import tqdm
from src.crnn import CRNN48
from src.national_id_dataset import NationalIdDataset

TOTAL_BATCHES = 3200
BATCH_SIZE = 512
CHECKPOINT_PATH = "drive/MyDrive/ocr/Checkpoints"


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_batch(model, criterion, optimizer, batch, target, target_lengths):
    preds = model.forward(batch)
    preds_size = torch.IntTensor([preds.size(0)] * BATCH_SIZE)
    cost = criterion(preds, target, preds_size, target_lengths)
    model.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == '__main__':
    model = CRNN48().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

    dataset = NationalIdDataset(batch_size=BATCH_SIZE)
    datasetIterator = iter(dataset)

    i = 0
    history = []
    for _ in tqdm(range(TOTAL_BATCHES)):
        x, y = next(datasetIterator)
        if i % 500 == 1:
            torch.save({
                "model": model.state_dict(),
                "Iteration": i,
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
        i += 1

    
    torch.save({
                "model": model.state_dict(),
                "Iteration": i,
                "Loss": cost,
                "Optimizer": optimizer.state_dict(),
                "History": history
            }, f"{CHECKPOINT_PATH}/{i}.torch")
