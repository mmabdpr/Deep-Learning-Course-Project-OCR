import torch
from torch import nn
from tqdm import tqdm
from crnn import CRNN
from national_id_dataset import NationalIdDataset
TOTAL_BATCHES = 20
BATCH_SIZE = 32
def train_batch(model, criterion, optimizer, batch, target, target_lengths):
    preds = model.forward(batch)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, target, preds_size, target_lengths) / batch_size
    model.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

model = CRNN()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CTCLoss(blank=0, zero_infinity=True)


datasetIterator = NationalIdDataset(batch_size=BATCH_SIZE)


for _ in tqdm(range(TOTAL_BATCHES)):
    x, y = next(datasetIterator)
    y_lengths = torch.full(size=(y.shape[0],), fill_value=y.shape[1], dtype=torch.long)
    cost = train_batch(model, criterion, optimizer, x, y, y_lengths)