def compute_offset_no_spaces(text, offset):
    count = 0
    for pos in range(offset):
        if text[pos] != ' ':
            count += 1
    return count


def count_length_no_special(text):
    count = 0
    special_char_list = [' ']
    for pos in range(len(text)):
        if text[pos] not in special_char_list:
            count += 1
    return count
