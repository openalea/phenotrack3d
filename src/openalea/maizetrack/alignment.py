import numpy as np
from copy import deepcopy
from scipy.spatial.distance import directed_hausdorff

from openalea.phenotracking.maize_track.utils import quantile_point


# def empty(row):
#    return all([math.isnan(x) for x in row])

# TODO : no maize/plant vocabulary

#######################################################################
# Functions for step 1 : mature leaf tracking
#######################################################################

# TODO : replace '-' by -1 ?
# TODO: add simple usage test (in test)
# TODO : keep local ?
def needleman_wunsch(X, Y, gap, local=False, gap_extremity_factor=1.):
    """
    Performs pairwise alignment of profiles X and Y with Needleman-Wunsch algorithm.
    A profile is defined as an array of one or more sequences of the same length.
    Each sequence includes one or several vectors of the same length.
    Source code : https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5
    The source code was modified to correct a few errors, and adapted to fit all requirements (extremity gap,
    customized scoring function, etc.)

    Parameters
    ----------
    X : array of shape (profile length, sequence length, vector length)
        profile 1
    Y : array of shape (profile length, sequence length, vector length)
        profile 2
    gap : float
        gap penalty parameter
    local : deprecated...
    gap_extremity_factor : float
        optional factor to increase/decrease gap penalty on sequence extremities

    Returns
    -------

    """
    if X.size == 0 and Y.size == 0:
        rx, ry = [], []
        return rx, ry

    gap_extremity = gap * gap_extremity_factor

    nx = X.shape[1]
    ny = Y.shape[1]

    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    if not local:
        F[:, 0] = np.linspace(start=0, stop=-nx*gap_extremity, num=nx+1)
        F[0, :] = np.linspace(start=0, stop=-ny*gap_extremity, num=ny+1)

    # Pointers to trace through an optimal alignment.
    P = np.zeros((nx + 1, ny + 1))
    P[:, 0] = 3
    P[0, :] = 4

    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):

            # TODO : gap argument should be useless ?
            t[0] = F[i, j] - scoring_function(X[:, i, :], Y[:, j, :], gap, gap_extremity)

            if j + 1 == ny:
                t[1] = F[i, j + 1] - gap_extremity
            else:
                t[1] = F[i, j + 1] - gap

            if i + 1 == nx:
                t[2] = F[i + 1, j] - gap_extremity
            else:
                t[2] = F[i + 1, j] - gap

            tmax = np.max(t)
            F[i + 1, j + 1] = tmax
            if t[0] == tmax:
                P[i + 1, j + 1] += 2
            if t[1] == tmax:
                P[i + 1, j + 1] += 3
            if t[2] == tmax:
                P[i + 1, j + 1] += 4

            if local and tmax < 0:
                F[i+1, j+1] = 0

    # Trace through an optimal alignment.
    if local:
        i, j = np.unravel_index(F.argmax(), F.shape)
    else:
        i, j = nx, ny

    rx, ry = [], []
    condition = True
    while condition:
        if P[i, j] in [2, 5, 6, 9]:
            rx.append(i - 1)
            ry.append(j - 1)
            i -= 1
            j -= 1
        elif P[i, j] in [3, 5, 7, 9]:
            rx.append(i - 1)
            ry.append('-')
            i -= 1
        elif P[i, j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(j - 1)
            j -= 1

        if local:
            condition = (i > 0 or j > 0) and F[i, j] > 0
        else:
            condition = i > 0 or j > 0

    rx = rx[::-1]
    ry = ry[::-1]

    return rx, ry


# def nw_score(X, Y, gap):
#     score = 0
#     if X.shape[1] == Y.shape[1]:
#         for i in range(X.shape[1]):
#             score += scoring_function(X[:, i, :], Y[:, i, :], gap, gap_extremity)
#     return score


def substitution_function(vec1, vec2, gap):
    """

    Compute a dissimilarity score between two vectors

    Parameters
    ----------
    vec1 : 1D array
    vec2 : 1D array, of same length than vec1
    gap :

    Returns
    -------

    """

    # TODO : check len == 3 if input = vec

    # should never happen. Only useful for nw_score()
    if all(np.isnan(vec1)) and all(np.isnan(vec2)):
        return gap * 2

    # same
    elif (all(np.isnan(vec1)) and not all(np.isnan(vec2))) or (all(np.isnan(vec2)) and not all(np.isnan(vec1))):
        return gap

    # height, length, azimut
    #weights = np.array([3, 1, 2])
    #weights = np.array([3, 1])
    #v1 = leaf1 * weights
    #v2 = leaf2 * weights
    #s = np.sqrt(np.sum((v1 - v2)**2))

    # TODO : specifique pour les features des feuilles du maïs, parametres a revoir
    dw = 3 * (vec1[0] - vec2[0])
    dl = 1 * (vec1[1] - vec2[1])
    da = abs(((vec1[2] - vec2[2]) + 1) % 2 - 1)
    s = np.sqrt(dw**2 + dl**2 + da**2)

    #s = 1 + abs(v1 - v2)
    #s[0] = s[0] ** 4
    #s -= 1

    return s


def scoring_function(x, y, gap, gap_extremity):
    """

    Compute a dissimilarity score between two columns of vectors x and y.

    Parameters
    ----------
    x
    y
    gap
    gap_extremity

    Returns
    -------

    """

    # TODO : gap argument should be useless ?

    # change 28/04, because it's not possible to calculate an azimuth mean :
    score = []
    for xvec in x:
        for yvec in y:
            if not all(np.isnan(xvec)) and not all(np.isnan(yvec)):
                score.append(substitution_function(xvec, yvec, gap))

    if score != []:
        score = np.mean(score)
    else:
        score = gap_extremity #bricolage

    return score


def insert_gaps(all_sequences, seq_indexes, alignment):
    """

    Add gaps in sequences of 'all_sequences' whose indexes is in 'seq_indexes'. A gap is defined as a NAN array element
    in a given sequence. Gaps positions are given by 'alignment'.

    Parameters
    ----------
    all_sequences
    seq_indexes
    alignment

    Returns
    -------

    """

    all_sequences2 = deepcopy(all_sequences)
    gap_indexes = [i for i, e in enumerate(alignment) if e == '-']

    vec_len = max([len(vec) for seq in all_sequences for vec in seq])

    for si in seq_indexes:
        for gi in gap_indexes:

            if all_sequences2[si].size == 0:
                all_sequences2[si] = np.full((1,vec_len), np.NAN)
            else:
                all_sequences2[si] = np.insert(all_sequences2[si], gi, np.NAN, 0)

    return all_sequences2


def multi_alignment(sequences, gap, gap_extremity_factor=1., direction=1, n_previous=5):
    """

    Alignment of n sequences.

    Parameters
    ----------
    sequences
    gap
    gap_extremity_factor
    direction
    n_previous

    Returns
    -------

    """

    aligned_sequences = deepcopy(sequences)

    start = int(-direction + 0.5)
    for k in range(start, len(aligned_sequences) + (start - 1)):

        xi = direction * np.arange(start, k + 1)  # ref
        yi = direction * (k + 1)

        # select the 2 motifs to align
        X = np.array([aligned_sequences[i] for i in xi[-n_previous:]])
        Y = np.array([aligned_sequences[yi]])

        # alignment
        rx, ry = needleman_wunsch(X, Y, gap, gap_extremity_factor=gap_extremity_factor)

        # update all sequences from sq0 to sq yi
        aligned_sequences = insert_gaps(aligned_sequences, xi, rx) # xi = sequences that all have already been aligned
        aligned_sequences = insert_gaps(aligned_sequences, [yi], ry)

    # convert list of aligned sequences (all having the same length) in a matrix of vector indexes (-1 = gap)
    # TODO : this matrix could be progressively constructed, in parallel to the alignment process ? Would be easier
    # TODO : to allow permutations for example..
    s = np.array(aligned_sequences).shape
    alignment_matrix = np.full((s[0], s[1]), -1)
    for i, aligned_seq in enumerate(aligned_sequences):
        no_gap = np.array([not all(np.isnan(e)) for e in aligned_seq])
        alignment_matrix[i][no_gap] = np.arange(sum(no_gap))

    return alignment_matrix


def detect_abnormal_ranks(alignment_matrix):
    """
    Algo specific to plant alignment.
    Detect abnormal columns in 'alignment_matrix' object resulting from multi alignment based on the following criteria:
    - A column is abnormal if it contains 2 times less aligned vectors in average (value != -1 in 'alignment_matrix')
    than the surrounding columns.
    - first and last columns can't be abnormal

    Parameters
    ----------
    alignment_matrix : 2D array
        result of multi_alignment() function

    Returns
    -------

    """

    alignment_matrix = np.array(alignment_matrix)
    counts = [len([k for k in alignment_matrix[:, i] if k != -1]) for i in range(alignment_matrix.shape[1])]
    abnormal_ranks = []
    for i in range(len(counts)):
        if 0 < i < len(counts) - 1 and counts[i] < 0.5 * np.mean([counts[i - 1], counts[i + 1]]):
            abnormal_ranks.append(i)

    return abnormal_ranks


#######################################################################
# Functions for step 2 : growing leaf tracking
#######################################################################

def polylines_distance(pl1, pl2, method):

    if method == 0:
        d1 = directed_hausdorff(pl1, pl2)[0]
        d2 = directed_hausdorff(pl2, pl1)[0]
        dist = max(d1, d2)

    # Frechet
    elif method == 1:

        dist = []
        n = 20
        for q in np.linspace(0, 1, n):
            pos1 = quantile_point(pl1, q)
            pos2 = quantile_point(pl2, q)
            dist.append(np.sqrt(np.sum((pos1 - pos2) ** 2)))
        dist = max(dist)

    elif method in [2, 3, 4, 5]:

        dist = 0
        n = 20
        for q in np.linspace(0, 1, n):
            pos1 = quantile_point(pl1, q)
            pos2 = quantile_point(pl2, q)
            dist += np.sqrt(np.sum((pos1 - pos2) ** 2))

        if method == 3:
            len1 = np.sum([np.linalg.norm(np.array(pl1[k]) - np.array(pl1[k + 1])) for k in range(len(pl1) - 1)])
            len2 = np.sum([np.linalg.norm(np.array(pl2[k]) - np.array(pl2[k + 1])) for k in range(len(pl2) - 1)])
            dist /= (min(len1, len2) / max(len1, len2))

        if method == 4:
            len1 = np.sum([np.linalg.norm(np.array(pl1[k]) - np.array(pl1[k + 1])) for k in range(len(pl1) - 1)])
            len2 = np.sum([np.linalg.norm(np.array(pl2[k]) - np.array(pl2[k + 1])) for k in range(len(pl2) - 1)])
            dist /= np.mean([len1, len2])

        if method == 5:
            len1 = np.sum([np.linalg.norm(np.array(pl1[k]) - np.array(pl1[k + 1])) for k in range(len(pl1) - 1)])
            len2 = np.sum([np.linalg.norm(np.array(pl2[k]) - np.array(pl2[k + 1])) for k in range(len(pl2) - 1)])
            dist /= np.max([len1, len2])

    return dist











