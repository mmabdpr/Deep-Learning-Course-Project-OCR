from pathlib import Path

import torch
import random
import numpy as np

from src.generate_national_cards import get_national_card_crop, get_national_card_crop_fit, get_national_card_crop_fit_rot, get_national_card_crop_from_template, get_random_national_id
from src.utils import id_label


class NationalIdDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, output_height):
        super(NationalIdDataset).__init__()
        self.batch_size = batch_size
        self.output_height = int(output_height)
        self.output_width = int(output_height / 90. * 360)

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
            card_ids = [get_random_national_id(
                with_hiphen=with_hiphen) for _ in range(n)]
            card_fonts = [fonts[i]
                          for i in np.random.choice(len(fonts), n, replace=True)]

            batch_x = []
            batch_y = []
            for i in range(n):
                card_id = card_ids[i]
                if random.random() < 0.6:
                    templates = [
                        Path(
                            'data/raw/national_card_crops/template_1.png').resolve().as_posix(),
                        Path(
                            'data/raw/national_card_crops/template_2.png').resolve().as_posix(),
                    ]
                    img = get_national_card_crop_from_template(
                        card_id,
                        np.random.choice(templates),
                        card_fonts[i][0],
                        card_fonts[i][1],
                        card_fonts[i][2],
                        (self.output_width, self.output_height))
                else:
                    if random.random() < 0.65:
                        if random.random() < 0.5:
                            img = get_national_card_crop_fit_rot(
                                card_id,
                                card_fonts[i][0],
                                card_fonts[i][1],
                                card_fonts[i][2],
                                48
                            )
                        else:
                            img = get_national_card_crop_fit(
                                card_id,
                                card_fonts[i][0],
                                card_fonts[i][1],
                                card_fonts[i][2],
                                48
                            )
                    img = get_national_card_crop(
                        card_id,
                        card_fonts[i][0],
                        card_fonts[i][1],
                        card_fonts[i][2],
                        (self.output_width, self.output_height))
                img = np.array(img)
                img = np.expand_dims(img, 0)

                batch_x.append(img)
                batch_y.append(id_label(card_id))

            max_width = max([x.shape[-1] for x in batch_x])
            batch_x = [np.pad(x, [(0, 0), (0, 0), (0, max_width - x.shape[-1])],
                              'constant', constant_values=0) for x in batch_x]

            batch_tensor_x = torch.tensor(
                np.array(batch_x, dtype=np.float32), dtype=torch.float32) / 255. - 0.5
            batch_tensor_y = torch.tensor(
                np.array(batch_y, dtype=np.int32), dtype=torch.int32)

            yield batch_tensor_x, batch_tensor_y


if __name__ == '__main__':
    ds = NationalIdDataset(32, 48)
    it = iter(ds)
    X, Y = next(it)
    X2, Y2 = next(it)
    pass
