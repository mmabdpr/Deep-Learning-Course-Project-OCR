import torch

import random
import numpy as np

import cv2

from src.generate_national_cards import get_national_card_crop, get_random_national_id, add_sp_noise, fa_to_en


def national_id_label(x):
    d = [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '-'
    ]
    
    en_x = fa_to_en(x)
    arr = [d.index(c) + 1 for c in en_x]
    arr = np.array(arr, dtype=np.int32)
    return arr


class NationalIdDataset(torch.utils.data.IterableDataset):
     def __init__(self, batch_size):
        super(NationalIdDataset).__init__()
        self.batch_size = batch_size
         
     def __iter__(self):
        while True:
            n = self.batch_size
            
            fonts = [
                ('fonts/bbadr.ttf', 48, 0),
                ('fonts/bmitra.ttf', 48, 0),
                ('fonts/bnazanin.ttf', 48, 0),
                ('fonts/byekan.ttf', 38, 10),
                ('fonts/btraffic.ttf', 42, 0),
            ]

            with_hiphen = random.random() > 0.5
            card_ids = [get_random_national_id(with_hiphen=with_hiphen) for _ in range(n)]
            card_fonts = [fonts[i]
                        for i in np.random.choice(len(fonts), n, replace=True)]
            kernel_sizes = np.random.randint(1, 6, n).tolist()

            batch_x = []
            batch_y = []
            for i in range(n):
                card_id = card_ids[i]
                img = get_national_card_crop(
                    card_id,
                    card_fonts[i][0],
                    card_fonts[i][1],
                    card_fonts[i][2],
                    (384, 96))
                img = np.array(img)[:, :, 0]
                img = cv2.erode(img,
                                np.ones((kernel_sizes[i], kernel_sizes[i]),
                                        np.uint8),
                                iterations=1)
                img = add_sp_noise(img)
                img = np.expand_dims(img, 0)  
                
                batch_x.append(img)
                batch_y.append(national_id_label(card_id))

            batch_tensor_x = torch.tensor(np.array(batch_x, dtype=np.float32), dtype=torch.float32) / 255. - 0.5
            batch_tensor_y = torch.tensor(np.array(batch_y, dtype=np.int32), dtype=torch.int32)
            
            yield batch_tensor_x, batch_tensor_y


    
if __name__ == '__main__':
    ds = NationalIdDataset(32)
    it = iter(ds)
    X, Y = next(it)
    X2, Y2 = next(it)
    pass