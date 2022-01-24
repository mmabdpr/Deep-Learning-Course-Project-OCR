import random
import numpy as np


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


def add_sp_noise(img, p=0.2):
    s_vs_p = 0.5
    amount = random.random() * p
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


def id_label(x):
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
    arr = [d.index(c) + 1 for c in en_x if c in d]
    arr = np.array(arr, dtype=np.int32)
    return arr
