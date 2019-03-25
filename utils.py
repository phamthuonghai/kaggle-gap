import numpy as np


def compute_offset_no_spaces(text, offset):
    count = 0
    for pos in range(offset):
        if text[pos] != ' ':
            count += 1
    return count


def count_chars_no_space(text):
    count = 0
    for pos in range(len(text)):
        if text[pos] != ' ':
            count += 1
    return count


def token_length_no_space(text):
    count = 0
    for pos in range(len(text)):
        if text[pos] != ' ':
            count += 1
    if text[:2] == '##':
        count -= 2
    return count


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
