from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageOps

import random
import numpy as np
from pathlib import Path

import cv2

from src.utils import add_sp_noise, fa_to_en


def get_credit_card_crop(card_text: str, font_path: str, font_size=48, font_y_offset=0, out_size=-1):

    fg_light = int(255 * random.random() * 0.4)
    bg_light = min(int(fg_light + 255 * (0.2 + random.random() * 0.4)), 255)

    fg = (0, 0, 0)
    bg = (bg_light, bg_light, bg_light)

    size = (500, 90)
    size2 = (400, 80)
    out_size = size if out_size == -1 else out_size

    rotation = random.random() * 10 - 5
    move_x = random.randint(0, 100)
    move_y = random.randint(5, 25)

    img = Image.new('RGB', size, bg)
    txt = Image.new('RGBA', size2, bg)

    d = ImageDraw.Draw(txt)
    font = ImageFont.truetype(font_path, font_size)
    d.text((0, 0), card_text, font=font, fill=fg, align='left')
    t = txt.rotate(rotation, expand=1)
    img.paste(t, (move_x, move_y - font_y_offset), t)
    img = img.resize(out_size)
    img = ImageOps.grayscale(img)

    img = np.array(img)
    kernel_size = random.randint(1, 3)
    img = cv2.erode(img,
                    np.ones((kernel_size, kernel_size),
                            np.uint8),
                    iterations=1)
    img = add_sp_noise(img)
    img = Image.fromarray(img)

    if random.random() > 0.5:
        img = ImageOps.invert(img)

    return img


def get_credit_card_crop_from_template(card_text: str, template_path: str, font_path: str, font_size=48, font_y_offset=0, out_size=-1):
    size = (500, 90)
    size2 = (490, 80)
    out_size = size if out_size == -1 else out_size

    rotation = random.random() * 10 - 5
    move_x = random.randint(0, 100)
    move_y = random.randint(5, 25)

    img = Image.open(template_path)

    img = np.array(img)
    img = img * ((128 + (255 * (random.random() * 0.2 - 0.1))) /
                 np.average(img))
    img_light = np.average(img)
    if img_light > (0.5 * 255):
        fg_light = int(img_light - 255 * (0.25 + random.random() * 0.25))
    else:
        fg_light = int(img_light + 255 * (0.25 + random.random() * 0.25))

    fg = (fg_light, fg_light, fg_light)

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)

    img = img.resize(size)
    txt = Image.new('RGBA', size2, (255, 255, 255, 0))

    d = ImageDraw.Draw(txt)
    font = ImageFont.truetype(font_path, font_size)
    W, H = img.size
    w, h = font.getsize(card_text)
    d.text((0, 0), card_text, font=font, fill=fg, align='left')
    t = txt.rotate(rotation, expand=1)
    img.paste(t, (move_x, move_y - font_y_offset), t)
    img = img.resize(out_size)
    img = ImageOps.grayscale(img)

    img = np.array(img)
    kernel_size = random.randint(1, 3)
    if fg_light < (0.5 * 255):
        img = cv2.erode(img,
                        np.ones((kernel_size, kernel_size),
                                np.uint8),
                        iterations=1)
    else:
        img = cv2.dilate(img,
                         np.ones((kernel_size, kernel_size),
                                 np.uint8),
                         iterations=1)
    img = add_sp_noise(img, 0.12)
    img = Image.fromarray(img)

    return img

# TODO remove space from label


def get_random_credit_id():
    id = np.random.choice(list('0123456789'), 16, replace=True).tolist()
    spaces = [4, 8, 12]

    res = []
    for i, c in enumerate(id):
        res.append(c)
        if (i + 1) in spaces:
            res.append(' ')

    res = ''.join(res)
    return res


def generate_random_credit_cards(n=1):
    res_dir = Path('data/interim/credit_id')

    # (name, size, y offset)
    # TODO add fonts
    fonts = [
        ('/home/mm/.local/share/fonts/kredit back.ttf', 42, 10),
        ('/home/mm/.local/share/fonts/kredit.ttf', 42, 10),
        ('/home/mm/.local/share/fonts/calibri.ttf', 42, 10),
        ('/usr/share/fonts/TTF/Inconsolata-Regular.ttf', 42, 10),
    ]

    card_ids = [get_random_credit_id() for _ in range(n)]
    card_fonts = [fonts[i]
                    for i in np.random.choice(len(fonts), n, replace=True)]

    for i in range(n):
        card_id = card_ids[i]
        if random.random() < .75:
            templates = [
                Path('data/raw/credit_crops/template_1.png').resolve().as_posix(),
                Path('data/raw/credit_crops/template_2.png').resolve().as_posix(),
                Path('data/raw/credit_crops/template_3.png').resolve().as_posix(),
                Path('data/raw/credit_crops/template_4.png').resolve().as_posix(),
            ]
            img = get_credit_card_crop_from_template(
                card_id,
                np.random.choice(templates),
                card_fonts[i][0],
                card_fonts[i][1],
                card_fonts[i][2],
                (267, 48))
        else:
            img = get_credit_card_crop(
                card_id,
                card_fonts[i][0],
                card_fonts[i][1],
                card_fonts[i][2],
                (267, 48))
        
        img.save(res_dir / f'{i}_{fa_to_en(card_id)}.jpg')


if __name__ == '__main__':
    generate_random_credit_cards(100)
