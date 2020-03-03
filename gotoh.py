# Author: Jose G. Perez <jperez50@miners.utep.edu>
import numpy as np
from timeit import default_timer as timer

from utils import get_input_float, get_input_boolean, combine

np.set_printoptions(threshold=np.inf, linewidth=500)
DEFAULT_A = 2
DEFAULT_B = 0.5
DEFAULT_MATCH_COST = 5
DEFAULT_MISMATCH_COST = -1

PATH_F_CHARACTERS = ['↑', '↓↓']
PATH_G_CHARACTERS = ['↖', '↓↓', '↑↑']
PATH_E_CHARACTERS = ['←', '↑↑']


def gotoh(S1, S2, match_cost, mismatch_cost, a, b):
    f = lambda k: a + (b * k)

    G = np.zeros((len(S2) + 1, len(S1) + 1))
    F = np.zeros_like(G)
    E = np.zeros_like(G)

    G_paths = np.zeros_like(G)
    G_paths[0, :] = 1
    G_paths[:, 0] = 2
    F_paths = np.zeros_like(F)
    E_paths = np.zeros_like(E)

    for j in range(1, len(S1) + 1):
        G[0, j] = E[0, j] = -1 * f(j)

    for i in range(1, len(S2) + 1):
        G[i, 0] = F[i, 0] = -1 * f(i)

    E[:, 0] = -np.inf
    F[0, :] = -np.inf

    for i in range(1, G.shape[0]):
        for j in range(1, G.shape[1]):
            # *************** F ***************
            f_choices = [F[i - 1, j] - b,
                         G[i - 1, j] - f(1)]
            f_idx = np.argmax(f_choices)
            F_paths[i, j] = f_idx
            F[i, j] = f_choices[f_idx]

            # *************** E ***************
            e_choices = [E[i, j - 1] - b,
                         G[i, j - 1] - f(1)]
            e_idx = np.argmax(e_choices)
            E_paths[i, j] = e_idx
            E[i, j] = e_choices[e_idx]

            # *************** G ***************
            if S1[j - 1] == S2[i - 1]:
                mcost = match_cost
            else:
                mcost = mismatch_cost

            g_choices = [G[i - 1, j - 1] + mcost,
                         E[i, j],
                         F[i, j]]
            g_idx = np.argmax(g_choices)
            G_paths[i, j] = g_idx
            G[i, j] = g_choices[g_idx]

    return F, E, G, F_paths, E_paths, G_paths


def reconstruct(S1, S2, F, E, G, F_paths, E_paths, G_paths):
    # Start in the bottom-right corner of G
    row_idx = G.shape[0] - 1
    col_idx = G.shape[1] - 1
    score = G[row_idx, col_idx]

    st1 = ''
    st2 = ''
    st1_idx = col_idx - 1
    st2_idx = row_idx - 1

    current_array = 'G'
    while row_idx > 0 and col_idx > 0:
        if current_array == 'G':
            direction = G_paths[row_idx, col_idx]
            diagonal_g = direction == 0
            jump_to_e = direction == 1
            jump_to_f = direction == 2

            if diagonal_g:
                st1 += S1[st1_idx]
                st2 += S2[st2_idx]
                st1_idx -= 1
                st2_idx -= 1

                row_idx -= 1
                col_idx -= 1
            elif jump_to_e:
                current_array = 'E'
                continue
            elif jump_to_f:
                current_array = 'F'
                continue

        elif current_array == 'E':
            direction = E_paths[row_idx, col_idx]
            left_e = direction == 0
            jump_to_g = direction == 1

            st1 += S1[st1_idx]
            st2 += '-'
            st1_idx -= 1
            col_idx -= 1

            if jump_to_g:
                current_array = 'G'
                continue
        else:
            direction = F_paths[row_idx, col_idx]
            up_f = direction == 0
            jump_to_g = direction == 1

            st1 += '-'
            st2 += S2[st2_idx]
            st2_idx -= 1
            row_idx -= 1

            if jump_to_g:
                current_array = 'G'
                continue

    # Reverse
    st1 = st1[::-1]
    st2 = st2[::-1]

    return st1, st2, score


if __name__ == '__main__':
    print(r"\
     ██████╗  ██████╗ ████████╗ ██████╗ ██╗  ██╗\
    ██╔════╝ ██╔═══██╗╚══██╔══╝██╔═══██╗██║  ██║\
    ██║  ███╗██║   ██║   ██║   ██║   ██║███████║\
    ██║   ██║██║   ██║   ██║   ██║   ██║██╔══██║\
    ╚██████╔╝╚██████╔╝   ██║   ╚██████╔╝██║  ██║\
     ╚═════╝  ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝\
    "
    )
    S1 = input('Type the first string (S1)')
    S2 = input('Type the second string (S2)')
    print("*** Default penalty costs and parameters for f(k) = a + (b * k) ***")
    print(f"\ta={DEFAULT_A}, b={DEFAULT_B}, Match Cost={DEFAULT_MATCH_COST}, Mismatch Cost={DEFAULT_MISMATCH_COST}")
    use_default = get_input_boolean('Do you want to use the above costs? [Y/N]')

    #%% Costs
    if use_default:
        print("* Using default costs *")
        a = DEFAULT_A
        b = DEFAULT_B
        match_cost = DEFAULT_MATCH_COST
        mismatch_cost = DEFAULT_MISMATCH_COST
    else:
        print("* Defining own costs *")
        a = get_input_float('Please type the value for a in f(k) = a + (b * k)')
        b = get_input_float('Please type the value for b in f(k) = a + (b * k)')
        match_cost = get_input_float('Please type the matching penalty/cost')
        mismatch_cost = get_input_float('Please type the mismatching penalty/cost')

    #%% Algorithm Timing
    start_time = timer()
    F, E, G, F_paths, E_paths, G_paths = gotoh(S1, S2, match_cost, mismatch_cost, a, b)
    st1, st2, score = reconstruct(S1, S2, F, E, G, F_paths, E_paths, G_paths)

    combined_F = combine(F, F_paths, PATH_F_CHARACTERS)
    combined_G = combine(G, G_paths, PATH_G_CHARACTERS)
    combined_E = combine(E, E_paths, PATH_E_CHARACTERS)

    end_time = timer()
    duration_sec = end_time - start_time

    #%% Results
    print("** Results **")

    print(f"Calculation took {duration_sec:.4f} seconds")
    print(f"Combined_F=\n{combined_F}")
    print(f"Combined_G=\n{combined_G}")
    print(f"Combined_E=\n{combined_E}")

    print(f"Original Inputs:")
    print(f"S1: {S1}")
    print(f"S2: {S2}")

    print(f"Alignment Score={score}")
    print(f"ST1: {st1}")
    print(f"ST2: {st2}")
