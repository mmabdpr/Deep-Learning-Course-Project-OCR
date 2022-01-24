from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageOps

import random
import numpy as np
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

import tqdm


# random.seed(42)

def get_national_card_crop(card_text: str, font_path: str, font_size=48, font_y_offset=0, out_size=-1):

    bg = (255, 255, 255)
    fg = (0, 0, 0)
    size = (360, 90)
    size2 = (340, 80)
    out_size = size if out_size == -1 else out_size

    rotation = random.random() * 10 - 5
    move_x = random.randint(-40, 40)
    move_y = random.randint(-30, 10)

    img = Image.new('RGB', size, bg)
    txt = Image.new('RGBA', size2, bg)

    d = ImageDraw.Draw(txt)
    font = ImageFont.truetype(font_path, font_size)
    W, H = img.size
    w, h = font.getsize(card_text)
    d.text(((W-w)/2, (H-h)/2), card_text, font=font, fill=fg)
    t = txt.rotate(rotation, expand=1)
    img.paste(t, (move_x, move_y - font_y_offset), t)
    img = img.resize(out_size)

    return img


def fa_to_en(text):
    d = {
        '۱': '1',
        '۲': '2',
        '۳': '3',
        '۴': '4',
        '۵': '5',
        '۶': '6',
        '۷': '7',
        '۸': '8',
        '۹': '9',
        '۰': '0',
    }

    return ''.join([d[x] if x in d.keys() else x for x in text])


def en_to_fa(text):
    d = {
        '1': '۱',
        '2': '۲',
        '3': '۳',
        '4': '۴',
        '5': '۵',
        '6': '۶',
        '7': '۷',
        '8': '۸',
        '9': '۹',
        '0': '۰',
    }

    return ''.join([d[x] if x in d.keys() else x for x in text])


def get_random_national_id(with_hiphen=True):
    id = np.random.choice(list('۱۲۳۴۵۶۷۸۹۰'), 10, replace=True).tolist()

    if with_hiphen:
        hiphens = np.random.choice(
            list(range(1, 9)), 2, replace=False).tolist()
    else:
        hiphens = []

    res = []
    for i, c in enumerate(id):
        res.append(c)
        if (i + 1) in hiphens:
            res.append('-')

    res = ''.join(res)
    return res


def add_sp_noise(img):
    s_vs_p = 0.5
    amount = random.random() * 0.3
    out = np.copy(img)

    # Salt
    num_salt = np.ceil(amount * img.size * s_vs_p)
    for x, y in zip(np.random.randint(0, img.shape[0], int(num_salt)),
                    np.random.randint(0, img.shape[1], int(num_salt))):
        out[x, y] = 255.

    # Pepper
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    for x, y in zip(np.random.randint(0, img.shape[0], int(num_pepper)),
                    np.random.randint(0, img.shape[1], int(num_pepper))):
        out[x, y] = 0.

    return out


def generate_random_national_cards(n=1):
    res_dir = Path('data/interim/national_id_nohiphens')

    # (name, size, y offset)
    fonts = [
        ('fonts/bbadr.ttf', 48, 0),
        ('fonts/bmitra.ttf', 48, 0),
        ('fonts/bnazanin.ttf', 48, 0),
        ('fonts/byekan.ttf', 38, 10),
        ('fonts/btraffic.ttf', 42, 0),
    ]

    card_ids = [get_random_national_id(False) for _ in range(n)]
    card_fonts = [fonts[i]
                  for i in np.random.choice(len(fonts), n, replace=True)]
    kernel_sizes = np.random.randint(1, 6, n).tolist()

    for i in tqdm.tqdm(range(n)):
        card_id = card_ids[i]
        img = get_national_card_crop(
            card_id,
            card_fonts[i][0],
            card_fonts[i][1],
            card_fonts[i][2])
        img = np.array(img)[:, :, 0]
        img = cv2.erode(img,
                        np.ones((kernel_sizes[i], kernel_sizes[i]),
                                np.uint8),
                        iterations=1)
        img = add_sp_noise(img)
        img = Image.fromarray(img)
        img.save(res_dir / f'{i}_{fa_to_en(card_id)}.jpg')

    # generate invert bw TODO


if __name__ == '__main__':
    generate_random_national_cards(3200)
