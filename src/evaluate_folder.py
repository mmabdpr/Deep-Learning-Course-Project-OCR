import torch
import numpy as np
from src.national_id_dataset import national_id_label
from src.metrics import calculate_accuracy
from src.decoder import ctc_decode
from src.train import BATCH_SIZE, device
from src.crnn import CRNN48
from tqdm import tqdm
from PIL import Image, ImageOps
import os

CHECKPOINT_PATH = "checkpoint\\13001.torch"
FOLDER_PATH = "data\\interim\\national_id_nohiphens"
ITERATIONS = 100
total_data_names = [f for f in os.listdir(
    FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))]


def get_batch(batch_index):
    X = []
    Y = []
    file_names = total_data_names[batch_index *
                                  BATCH_SIZE: (batch_index + 1) * BATCH_SIZE]
    for fn in file_names:
        path = os.path.join(FOLDER_PATH, fn)
        data = Image.open(path)
        data = ImageOps.grayscale(data)
        data = data.resize((384, 96))
        data = np.asarray(data, dtype=float) / 255 - 0.5
        data = np.expand_dims(data, 0)
        # data = np.expand_dims(data, 0)
        data = np.float32(data)
        X.append(torch.tensor(data))
        label = fn.split('.')[0].split('_')[1]
        y = national_id_label(label)
        Y.append(torch.tensor(y))
    return torch.stack(X), torch.stack(Y)


if __name__ == "__main__":
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = CRNN48()
    model.load_state_dict(checkpoint['model'])
    overallAccuarcy = 0
    model.eval()
    idx = 0
    get_batch(3)
    for _ in tqdm(range(ITERATIONS)):
        x, y = get_batch(idx)
        x.to(device)
        y.to(device)
        with torch.no_grad():
            pred1 = model.forward(x)
        pred2 = ctc_decode(pred1, beam_size=10)
        overallAccuarcy += calculate_accuracy(y, pred2)
        idx += 1
    print(overallAccuarcy / ITERATIONS)
