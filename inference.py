from src.crnn import CRNN
import torch
from PIL import Image, ImageOps
import numpy as np
from decoder import ctc_decode
label2char = {0:'', 1:"0", 2: "1", 3: "2", 4: "3", 5: "4", 6: "5", 7: "6", 8: "7", 9: "8", 10: "9", 11: "-"}
if __name__ == "__main__":
    model = CRNN()
    checkpoint = torch.load("checkpoint\\12001.torch")
    model.load_state_dict(checkpoint['model'])
    data = "data\\national_id\\0_3-053897-904.jpg"
    data = Image.open(data)
    data = ImageOps.grayscale(data)
    data = np.asarray(data, dtype=float) / 255 - 0.5
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, 0)
    data = np.float32(data)
    pred = model(torch.tensor(data))
    res = ctc_decode(pred, label2char=label2char, beam_size=4)

