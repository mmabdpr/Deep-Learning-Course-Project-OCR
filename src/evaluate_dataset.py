from pathlib import Path
import torch
import numpy as np
from src.national_id_dataset import NationalIdDataset
from src.metrics import calculate_accuracy
from src.decoder import ctc_decode
from src.train import BATCH_SIZE, device
from src.crnn import CRNN48
from tqdm import tqdm


CHECKPOINT_PATH = (Path("checkpoint") / "13001.torch").resolve().as_posix()
ITERATIONS = 100


if __name__ == "__main__":
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = CRNN48()
    model.load_state_dict(checkpoint['model'])
    dataset = NationalIdDataset(batch_size=BATCH_SIZE)
    datasetIterator = iter(dataset)
    overallAccuarcy = 0
    model.eval()
    for _ in tqdm(range(ITERATIONS)):
        x, y = next(datasetIterator)
        x.to(device)
        y.to(device)
        with torch.no_grad():
            pred1 = model.forward(x)
        pred2 = ctc_decode(pred1, beam_size=10)
        overallAccuarcy += calculate_accuracy(y, pred2)
    print(overallAccuarcy / ITERATIONS)
