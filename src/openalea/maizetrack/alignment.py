import numpy as np
from copy import deepcopy
from openalea.maizetrack.utils import quantile_point, polyline_until_z

##########################################################################################################
##### Functions for step 1 : mature leaf tracking ########################################################
##########################################################################################################


# TODO : replace '-' by -1 ?
def needleman_wunsch(X, Y, gap, gap_extremity_factor=1.):
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
            t[0] = F[i, j] - alignment_score(X[:, i, :], Y[:, j, :], gap_extremity)

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

    # Trace through an optimal alignment.
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

        condition = i > 0 or j > 0

    rx = rx[::-1]
    ry = ry[::-1]

    return rx, ry


def scoring_function(vec1, vec2):
    """

    Compute a dissimilarity score between two vectors of same length, which is equal to their euclidian distance.

    Parameters
    ----------
    vec1 : 1D array
    vec2 : 1D array, of same length than vec1
    gap : float

    Returns
    -------
    float

    """

    return np.linalg.norm(vec1 - vec2)


def alignment_score(x, y, gap_extremity):
    """

    Compute a dissimilarity score between two arrays of vectors x and y.
    x and y can have different lengths, but all vectors in x and y must have the same length.

    Parameters
    ----------
    x : 2D array
        size (number of vectors, vector length)
    y : 2D array
        size (number of vectors, vector length)
    gap_extremity : float

    Returns
    -------
    float

    """

    # list of scores for each pair of vectors xvec and yvec,
    # with xvec and yvec being non-gap elements of x and y respectively.
    score = []
    for xvec in x:
        for yvec in y:
            if not all(np.isnan(xvec)) and not all(np.isnan(yvec)):
                score.append(scoring_function(xvec, yvec))

    if score:
        score = np.mean(score)
    else:
        score = gap_extremity # TODO : bricolage

    return score


def insert_gaps(all_sequences, seq_indexes, alignment):
    """

    Add gaps in sequences of 'all_sequences' whose indexes is in 'seq_indexes'. A gap is defined as a NAN array element
    in a given sequence. Gaps positions are given by 'alignment'.

    Parameters
    ----------
    all_sequences : list of 2D arrays
    seq_indexes : list of int
    alignment : list of int and str ('-')
        result from needleman_wunsch()

    Returns
    -------

    """

    all_sequences2 = deepcopy(all_sequences)
    gap_indexes = [i for i, e in enumerate(alignment) if e == '-']

    vec_len = max([len(vec) for seq in all_sequences for vec in seq])

    for si in seq_indexes:
        for gi in gap_indexes:

            if all_sequences2[si].size == 0:
                all_sequences2[si] = np.full((1, vec_len), np.NAN)
            else:
                all_sequences2[si] = np.insert(all_sequences2[si], gi, np.NAN, 0)

    return all_sequences2


def multi_alignment(sequences, gap, gap_extremity_factor=1., direction=1, n_previous=0):
    """

    Multi sequence alignment algorithm to align n sequences, using a progressive method. At each step, a sequence (Y)
    is aligned with a matrix (X) corresponding to the alignement of k sequences, resulting in the alignment of
    k + 1 sequences. Each pairwise alignment of X vs Y is based on needleman-wunsch algorithm.

    Parameters
    ----------
    sequences : list of 2D arrays
        The list of sequences to align
    gap : float
        penalty parameter to add a gap
    gap_extremity_factor : float
        parameter to modify the gap penalty on sequence extremity positions, relatively to gap value.
        For example, if gap = 5 and gap_extremity_factor = 0.6, Then the penalty for terminal gaps is equal to 3.
    direction : int
        if direction == 1 : align from t=1 to t=tmax. If direction == -1 : align from t=tmax to t=1.
    n_previous : int


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
    # TODO : this matrix could be progressively constructed, in parallel to the alignment process. It would be easier
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

def polylines_distance(pl1, pl2, n=20):
    """ compute the distance between two polylines """

    dist = 0
    for q in np.linspace(0, 1, n):
        pos1 = quantile_point(pl1, q)
        pos2 = quantile_point(pl2, q)
        dist += np.sqrt(np.sum((pos1 - pos2) ** 2))

    return dist


def phm_leaves_distance(leaf_ref, leaf_candidate, method):
    """ takes two leaf objects, deduct two polylines which start from a same point, compute the distance between
     these two polylines """

    # creating two polylines starting from the same base
    leaf1, leaf2 = leaf_ref, leaf_candidate
    pl1 = leaf1.real_pl
    pl2 = leaf2.real_pl
    zbase1, zbase2 = pl1[0][2], pl2[0][2]
    if zbase1 < zbase2:
        pl2 = polyline_until_z(leaf2.highest_pl, zbase1)
    else:
        pl1 = polyline_until_z(leaf1.highest_pl, zbase2)

    # normalize
    len1 = np.sum([np.linalg.norm(np.array(pl1[k]) - np.array(pl1[k + 1])) for k in range(len(pl1) - 1)])
    pl1 = pl1 / np.max((len1, 0.0001))
    pl2 = pl2 / np.max((len1, 0.0001))

    # computing distance
    d = polylines_distance(pl1, pl2, method)

    return d