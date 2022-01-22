import torch
from torch import nn
from tqdm import tqdm
from src.crnn import CRNN
from src.national_id_dataset import NationalIdDataset

TOTAL_BATCHES = 20
BATCH_SIZE = 32


def train_batch(model, criterion, optimizer, batch, target, target_lengths):
    preds = model.forward(batch)
    preds_size = torch.IntTensor([preds.size(0)] * BATCH_SIZE)
    cost = criterion(preds, target, preds_size, target_lengths)
    model.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == '__main__':
    model = CRNN()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')


    dataset = NationalIdDataset(batch_size=BATCH_SIZE)
    datasetIterator = iter(dataset)


    for _ in tqdm(range(TOTAL_BATCHES)):
        x, y = next(datasetIterator)
        y_lengths = torch.full(
            size=(y.shape[0],), fill_value=y.shape[1], dtype=torch.int32
        )
        cost = train_batch(model, criterion, optimizer, x, y, y_lengths)
