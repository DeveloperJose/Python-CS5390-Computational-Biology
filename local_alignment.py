# Author: Jose G. Perez <jperez50@miners.utep.edu>
import numpy as np
from timeit import default_timer as timer

from utils import get_input_float, get_input_boolean, combine

np.set_printoptions(threshold=np.inf, linewidth=500)
DEFAULT_INS_COST = -0.5
DEFAULT_DEL_COST = -0.5
DEFAULT_MATCH_COST = 5
DEFAULT_MISMATCH_COST = -1
PATH_CHARACTERS = ['', '↖', '↑', '←']


#%% Algorithm: Smith-Waterman
def smith_waterman(S1, S2, ins_cost, del_cost, match_cost, mismatch_cost):
    V = np.zeros((len(S2) + 1, len(S1) + 1))
    paths = np.zeros_like(V)
    V[:, 0] = V[0, :] = 0

    for i in range(1, V.shape[0]):
        for j in range(1, V.shape[1]):
            # Check if it's a match or mismatch
            if S1[j - 1] == S2[i - 1]:
                mcost = match_cost
            else:
                mcost = mismatch_cost

            choices = [0,
                       V[i-1, j-1] + mcost,
                       V[i-1, j] + del_cost,
                       V[i, j-1] + ins_cost]

            idx = np.argmax(choices)
            paths[i, j] = idx
            V[i, j] = choices[idx]
    return V, paths


def reconstruct(S1, S2, V, paths):
    # For local alignment, pick the highest number in V
    maxValue = np.amax(V)
    idxs = np.where(V == maxValue)
    start_row_idx = idxs[0][0]
    start_col_idx = idxs[1][0]

    # Then backtrack until we hit a 0
    # We also keep backtracking until we reach the top-left corner
    st1 = ''
    st2 = ''
    row_idx = start_row_idx
    col_idx = start_col_idx
    st1_idx = col_idx - 1
    st2_idx = row_idx - 1
    score = V[row_idx, col_idx]
    while row_idx > 0 or col_idx > 0:
        value = paths[row_idx, col_idx]
        diagonal = (value == 1) or (value == 0 and col_idx >= 1 and row_idx >= 1)
        up = (value == 2) or (value == 0 and row_idx >= 1)
        left = (value == 3) or (value == 0 and col_idx >= 1)
        if diagonal:
            row_idx -= 1
            col_idx -= 1

            st1 += S1[st1_idx]
            st2 += S2[st2_idx]
            st1_idx -= 1
            st2_idx -= 1
        elif up:
            row_idx -= 1

            st1 += '-'
            st2 += S2[st2_idx]
            st2_idx -= 1
        elif left:
            col_idx -= 1

            st1 += S1[st1_idx]
            st2 += '-'
            st1_idx -= 1

    # Reverse
    st1 = st1[::-1]
    st2 = st2[::-1]

    # Add missing right side (go from max spot to bottom-right corner)
    row_idx = start_row_idx
    col_idx = start_col_idx
    st1_idx = col_idx
    st2_idx = row_idx

    while row_idx <= len(S1) and col_idx <= len(S2):
        down = (row_idx + 1 <= len(S2))
        right = (col_idx + 1 <= len(S1))
        diagonal = (down and right)

        if diagonal:
            row_idx += 1
            col_idx += 1

            st1 += S1[st1_idx]
            st2 += S2[st2_idx]
            st1_idx += 1
            st2_idx += 1
        elif down:
            row_idx += 1

            st1 += '-'
            st2 += S2[st2_idx]
            st2_idx += 1
        elif right:
            col_idx += 1

            st1 += S1[st1_idx]
            st2 += '-'
            st1_idx += 1
        else:
            break

    return st1, st2, score


if __name__ == '__main__':
    #%% User Input
    print(r"\
    ███████╗███╗   ███╗██╗████████╗██╗  ██╗      ██╗    ██╗ █████╗ ████████╗███████╗██████╗ ███╗   ███╗ █████╗ ███╗   ██╗ \
    ██╔════╝████╗ ████║██║╚══██╔══╝██║  ██║      ██║    ██║██╔══██╗╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██╔══██╗████╗  ██║ \
    ███████╗██╔████╔██║██║   ██║   ███████║█████╗██║ █╗ ██║███████║   ██║   █████╗  ██████╔╝██╔████╔██║███████║██╔██╗ ██║ \
    ╚════██║██║╚██╔╝██║██║   ██║   ██╔══██║╚════╝██║███╗██║██╔══██║   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══██║██║╚██╗██║ \
    ███████║██║ ╚═╝ ██║██║   ██║   ██║  ██║      ╚███╔███╔╝██║  ██║   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║ \
    ╚══════╝╚═╝     ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝       ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝")

    S1 = input('Type the first string (S1)')
    S2 = input('Type the second string (S2)')

    print("* Default penalty costs *")
    print(f"\tInsertion Cost={DEFAULT_INS_COST}, Deletion Cost={DEFAULT_DEL_COST}, Match Cost={DEFAULT_MATCH_COST}, Mismatch Cost={DEFAULT_MISMATCH_COST}")
    use_default = get_input_boolean('Do you want to use the above costs? [Y/N]')

    #%% Costs
    if use_default:
        print("* Using default costs *")
        ins_cost = DEFAULT_INS_COST
        del_cost = DEFAULT_DEL_COST
        match_cost = DEFAULT_MATCH_COST
        mismatch_cost = DEFAULT_MISMATCH_COST
    else:
        print("* Defining own costs *")
        ins_cost = get_input_float('Please type the insertion penalty/cost')
        del_cost = get_input_float('Please type the deletion penalty/cost')
        match_cost = get_input_float('Please type the matching penalty/cost')
        mismatch_cost = get_input_float('Please type the mismatching penalty/cost')

    #%% Algorithm Timing
    start_time = timer()
    V, paths = smith_waterman(S1, S2, ins_cost, del_cost, match_cost, mismatch_cost)
    st1, st2, score = reconstruct(S1, S2, V, paths)

    combined = combine(V, paths, PATH_CHARACTERS)

    end_time = timer()
    duration_sec = end_time - start_time

    #%% Results
    print("** Results **")
    print(f"\tFor: Insertion Cost={ins_cost}, Deletion Cost={del_cost}, Match Cost={match_cost}, Mismatch Cost={mismatch_cost}")

    print(f"Calculation took {duration_sec:.4f} seconds")
    print(f"V=\n{V}")
    print(f"Paths=\n{paths}")
    print(f"Combined=\n{combined}")

    print(f"Original Inputs:")
    print(f"S1: {S1}")
    print(f"S2: {S2}")

    print(f"Alignment Score={score}")
    print(f"ST1: {st1}")
    print(f"ST2: {st2}")