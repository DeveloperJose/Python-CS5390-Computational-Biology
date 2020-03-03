# Author: Jose G. Perez <jperez50@miners.utep.edu>
import numpy as np
from timeit import default_timer as timer
from global_alignment import needleman_wunsch, reconstruct
from utils import get_input_float, get_input_boolean, get_input_int

DEFAULT_INS_COST = -0.5
DEFAULT_DEL_COST = -0.5
DEFAULT_MATCH_COST = 5
DEFAULT_MISMATCH_COST = -1


#%% Functions
def combine_sequences(seq_star_list, seq_list):
    combined_star = ''
    combined_seq_list = [''] * len(seq_list)
    idxs = [0] * len(seq_list)

    while True:
        # Stop running when all the sequences are aligned
        isRunning = True
        for seq_list_idx, idx in enumerate(idxs):
            seq = seq_list[seq_list_idx]
            seq_len = len(seq)
            if idx >= seq_len:
                break

        if not isRunning:
            break

        # Compare the star sequences, check if one of the star sequences has a gap
        hasGap = False
        for seq_star_num, seq_star in enumerate(seq_star_list):
            seq_star_char_idx = idxs[seq_star_num]
            try:
                next_char = seq_star[seq_star_char_idx]
            except IndexError:
                return combined_star, combined_seq_list
            if next_char == '-':
                # print(f'Found a gap: {seq_star_num, seq_star}')
                hasGap = True
                idxs[seq_star_num] += 1
                break

        # There was a gap in one, add a gap to the NORMAL sequences
        if hasGap:
            combined_star += '-'
            for seq_num in range(len(combined_seq_list)):
                combined_seq_list[seq_num] += '-'
        # There was no gap in any, so copy the character of each one to their respective sequences
        else:
            for seq_num in range(len(combined_seq_list)):
                seq1 = seq_list[seq_num]
                idx1 = idxs[seq_num]
                combined_seq_list[seq_num] += seq1[idx1]
                idxs[seq_num] += 1

            combined_star += seq_star_list[seq_num][idx1]

    return combined_star, combined_seq_list


#%% Main
print(r"\
 ██████╗███████╗███╗   ██╗████████╗███████╗██████╗     ███████╗████████╗ █████╗ ██████╗\
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔════╝██╔══██╗    ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗\
██║     █████╗  ██╔██╗ ██║   ██║   █████╗  ██████╔╝    ███████╗   ██║   ███████║██████╔╝\
██║     ██╔══╝  ██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗    ╚════██║   ██║   ██╔══██║██╔══██╗\
╚██████╗███████╗██║ ╚████║   ██║   ███████╗██║  ██║    ███████║   ██║   ██║  ██║██║  ██║\
 ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝\
"
)

print("* Default penalty costs *")
print(f"\tInsertion Cost={DEFAULT_INS_COST}, Deletion Cost={DEFAULT_DEL_COST}, Match Cost={DEFAULT_MATCH_COST}, Mismatch Cost={DEFAULT_MISMATCH_COST}")
use_default = get_input_boolean('Do you want to use the above costs? [Y/N]')
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

s_list = []
s_count = get_input_int('How many sequences will you align?')
for idx in range(s_count):
    next_seq = input(f'Input sequence #{idx+1}')
    s_list.append(next_seq)

select_center = get_input_boolean('Do you want to input the center star sequence? [Y/N]')
if select_center:
    center_star_seq = input('Input the center star sequence')
    center_star_st1s = []
    center_star_st2s = []
    for seq in s_list:
        if seq == center_star_seq:
            continue

        V, paths = needleman_wunsch(center_star_seq, seq, ins_cost, del_cost, match_cost, mismatch_cost)
        st1, st2, score = reconstruct(center_star_seq, seq, V, paths)
        center_star_st1s.append(st1)
        center_star_st2s.append(st2)
else:
    print("Finding center star...")
    start_time_total = timer()
    best_d = np.inf
    center_star_seq = ''
    center_star_st1s = []
    center_star_st2s = []
    for idx1, seq1 in enumerate(s_list):
        d_sum = 0
        st1_list = []
        st2_list = []
        for idx2, seq2 in enumerate(s_list):
            # Don't match
            if idx1 == idx2:
                continue

            V, paths = needleman_wunsch(seq1, seq2, ins_cost, del_cost, match_cost, mismatch_cost)
            st1, st2, score = reconstruct(seq1, seq2, V, paths)
            st1_list.append(st1)
            st2_list.append(st2)
            d_sum += score

        if d_sum < best_d:
            best_d = d_sum
            center_star_seq = seq1
            center_star_st1s = st1_list
            center_star_st2s = st2_list

    end_time_star = timer()
    duration_sec = end_time_star - start_time_total
    print(f"Finding center star took {duration_sec:.4f} seconds")
    print(f"Center Star Sequence: {center_star_seq}")

#%% Combine
combined_star, combined_seq_list = combine_sequences(center_star_st1s, center_star_st2s)

# Clean up
idxs_to_erase = []
for char_idx, char in enumerate(combined_star):
    gap_count = 0
    for seq in combined_seq_list:
        if seq[char_idx] == '-':
            gap_count += 1

    if gap_count == len(combined_seq_list):
        idxs_to_erase.append(char_idx)

combined_star = ''.join([char for idx, char in enumerate(combined_star) if idx not in idxs_to_erase])
for seq_idx in range(len(combined_seq_list)):
    seq = combined_seq_list[seq_idx]
    combined_seq_list[seq_idx] = ''.join([char for idx, char in enumerate(seq) if idx not in idxs_to_erase])

print("* Alignment *")
print(f"Sequence: {combined_star} <- Center Star")
for st in combined_seq_list:
    print(f"Sequence: {st}")