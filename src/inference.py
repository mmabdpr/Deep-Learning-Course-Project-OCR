from src.crnn import CRNN
import torch
from PIL import Image, ImageOps
import numpy as np
from src.decoder import ctc_decode
from src.metrics import calculate_accuracy


label2char = {0: '', 1: "0", 2: "1", 3: "2", 4: "3", 5: "4",
              6: "5", 7: "6", 8: "7", 9: "8", 10: "9", 11: "-"}


if __name__ == "__main__":
    model = CRNN()
    checkpoint = torch.load("checkpoint\\13001.torch")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    data = "data\\interim\\national_id\\1037_75-861964-26.jpg"
    data = Image.open(data)
    data = ImageOps.grayscale(data)
    data = data.resize((384, 96))
    data = np.asarray(data, dtype=float) / 255 - 0.5
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, 0)
    data = np.float32(data)
    pred = model(torch.tensor(data))
    res = ctc_decode(pred, label2char=None, beam_size=4)
    res = torch.tensor(np.array(res))
    label = torch.tensor(np.array([[4, 11, 1, 6, 4, 9, 10, 8, 11, 10, 1, 5]]))
    calculate_accuracy(label, res)
    print(''.join(res[0]))
