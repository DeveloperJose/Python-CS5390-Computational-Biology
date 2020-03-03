# Author: Jose G. Perez <jperez50@miners.utep.edu>
import numpy as np


def get_input_float(message):
    while True:
        try:
            return float(input(message))
        except ValueError:
            print('Input was not a float!')
            continue


def get_input_int(message):
    while True:
        try:
            return int(input(message))
        except ValueError:
            print('Input was not an int!')
            continue


def get_input_boolean(message):
    while True:
        st = input(message).upper()
        if "Y" in st:
            return True
        elif "N" in st:
            return False

        print("Please type Y or N")
        continue


def combine(arr, path, chars):
    combined = np.zeros_like(arr).astype(np.object)
    for row_idx, (value_row, direction_row) in enumerate(zip(arr, path)):
        for col_idx, (value, direction) in enumerate(zip(value_row, direction_row)):
            if value == np.inf:
                combined[row_idx, col_idx] = '∞'
            elif value == -np.inf:
                combined[row_idx, col_idx] = '-∞'
            else:
                combined[row_idx, col_idx] = chars[int(direction)] + str(value)
    return combined