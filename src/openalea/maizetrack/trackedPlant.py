"""
Classes for maize leaf tracking and rank attribution, in a time-series of Phenomenal 3D segmentation objects.
//!\\ In the tracking algorithm, ranks start at 0. But in the final output (stored in the 'pm_leaf_number_tracking'
attribute of each leaf), ranks start at 1. (see get_ranks() method)
"""

import numpy as np
import warnings

from openalea.maizetrack.alignment import multi_alignment
from openalea.maizetrack.alignment_postprocessing import detect_abnormal_ranks, leaf_polylines_distance


def check_time_intervals(times, discontinuity=5.):
    dt_median = np.median(np.diff(times))
    valid = np.array([True] * len(times))
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]) > discontinuity * dt_median:
            valid[i:] = False
    return valid


class TrackedLeaf:
    """
    Describe a leaf organ, with attributes specific to leaf tracking algorithm.
    """

    def __init__(self, polyline, features):

        self.polyline = polyline
        self.features = features

        self.vec = np.array([])

    def compute_features_vector(self, w_h, w_l):
        """ for sequence alignment of mature leaves """

        if not self.features['mature']:
            warnings.warn('This method is supposed to be used for mature leaves')

        azimuth_scaled = self.features['azimuth'] / 360 * 2 * np.pi  # [0, 360] interval --> [-1, 1] interval

        self.vec = np.array([np.cos(azimuth_scaled),
                             np.sin(azimuth_scaled),
                             w_h * self.features['height'],
                             w_l * self.features['length']])


class TrackedSnapshot:
    """
    Describe the plant segmentation from Phenomenal at a given time point, and associate it to its corresponding
    metainfos. Describe the order of leaves, which is modified during leaf tracking.
    """

    def __init__(self, leaves, check):

        self.leaves = leaves

        self.check_continuity = check

        # self.sequence gives the ranks of leaves in self.leaves.
        # for example, if self.order[5] = 2, it means that self.leaves[2] is associated to rank 5+1=6.
        # -1 = no leaf
        self.sequence = []

    def leaf_ranks(self):
        """
        list of ranks of leaves contained in self.leaves

        small example :
        self.leaves = [leaf0, leaf1, leaf2, leaf3]
        self.sequence = [-1, -1, 0, 1, -1, 2, 3, -1]
        ===> self.leaf_ranks() returns [3, 4, 6, 7]

        WARNING : the rank of a leaf is given by its position in TrackedSnapshot.sequence, which starts at 0.
        But leaf ranks are usually numerated starting from 1: this second option is used in the output from this
        function.
        """
        return [self.sequence.index(i) + 1 if i in self.sequence else 0 for i in range(len(self.leaves))]


class TrackedPlant:
    """
    Class describing a time-series of TrackedSnapshot. It is used to track maize leaves over time and find their ranks.
    """

    def __init__(self, snapshots):

        # list of TrackedSnapshot objects
        self.snapshots = snapshots

    # ===== method to load a TrackedPlant object ====================================================================

    @staticmethod
    def load(segmentation_time_series):
        """
        Parameters
        ----------
        segmentation_time_series : list
            list of dict {'time': float,
                          'polylines_sequence': list of polylines,
                          'features_sequence': {'mature': bool, 'azimuth': float, 'height': float, 'length': float}
                          }

        Returns
        -------
        """

        times = [seg['time'] for seg in segmentation_time_series]

        # verify temporal order of the time-series
        if times != sorted(times):
            raise Exception('objects need to be ordered by temporal order')

        # check if there is no big time gap in the time-series
        checks_continuity = check_time_intervals(times)

        # initialize the TrackedPlant object
        snapshots = []
        for seg, check in zip(segmentation_time_series, checks_continuity):
            leaves = []
            for polyline, features in zip(seg['polylines_sequence'], seg['features_sequence']):
                assert(all(var in features for var in ['mature', 'azimuth', 'height', 'length']))
                leaves.append(TrackedLeaf(polyline=polyline, features=features))
            snapshots.append(TrackedSnapshot(leaves, check))

        return TrackedPlant(snapshots=snapshots)

    def get_ref_skeleton(self, nmax=15):
        """
        Compute a median skeleton {rank : leaf}.
        For each rank, the leaf whose vector is less distant to all other leaves from the same ranks is selected.

        Parameters
        ----------
        nmax : int
            max number of leaves considered at a given rank (to avoid old leaves which can have senescence)

        Returns
        -------

        """

        ref_skeleton = dict()

        ranks = range(len(self.snapshots[0].sequence))
        for rank in ranks:
            # all matures leaves for this rank
            leaves = [s.leaves[s.sequence[rank]] for s in self.snapshots if s.sequence[rank] != -1]  # -1 = no leaf
            leaves = [leaf for leaf in leaves if leaf.features['mature']]

            # remove old leaves (that could have a different shape)
            leaves = leaves[:nmax]

            if len(leaves) > 0:
                vectors = np.array([leaf.vec for leaf in leaves])
                mean_vector = np.mean(vectors, axis=0)
                dists = [np.sum(abs(vec - mean_vector)) for vec in vectors]
                ref_skeleton[rank] = leaves[np.argmin(dists)]

        return ref_skeleton

    def mature_leaf_tracking(self, gap=12.35, gap_extremity_factor=0.2, start=0, w_h=0.03, w_l=0.004, align_range=None,
                             rank_attribution=True):
        """
        alignment and rank attributions in a time-series of sequences of leaves.
        Step 1 : use a multiple sequence alignment algorithm to align the sequences.
        Step 2 : Detect and remove abnormal group of leaves ; final rank attribution.

        Parameters
        ----------
        gap : float
            weight  for pairwise sequence alignment
        gap_extremity_factor : float
            parameter allowing to change the value of the gap penalty for terminal gaps (terminal gap penalty = gap *
            gap_extremity_factor)
        direction : int
            parameter determining if sequences are progressively aligned starting with first (direction == 1) or last
            (direction == -1) sequences from the time-series.
        n_previous : int
            Each time a new sequence is added to the global alignment, it is compared with the n_previous previous
            sequences added to the global alignment.
        w_h : float
            weight associated to insertion height feature in a leaf feature vector
        w_l : float
            weight associated to length feature in a leaf feature vector
        rank_attribution : bool
            choose if step 2 is done (True) or not (False)

        Returns
        -------

        """

        # ===== Step 1 - multi sequence alignment =====================================================================

        # initialize sequence attribute of each snapshot, with only mature leaves:
        for snapshot in self.snapshots:
            snapshot.sequence = [i for i, leaf in enumerate(snapshot.leaves) if leaf.features['mature']]

        # compute features vectors for mature leaves
        for snapshot in self.snapshots:
            for leaf in snapshot.leaves:
                if leaf.features['mature']:
                    leaf.compute_features_vector(w_h=w_h, w_l=w_l)

        # time-series of sequences of features vectors (sequences have different sizes, vectors have the same size)
        features_sequences = []
        for snapshot in self.snapshots:
            seq = np.array([snapshot.leaves[i].vec for i in snapshot.sequence])
            features_sequences.append(seq)

        # sequence alignment
        alignment_matrix = multi_alignment(sequences=features_sequences,
                                           gap=gap, gap_extremity_factor=gap_extremity_factor,
                                           align_range=align_range, start=start)

        # update sequence attributes
        for t, aligned_sequence in enumerate(alignment_matrix):
            self.snapshots[t].sequence = [-1 if i == -1 else self.snapshots[t].sequence[i] for i in aligned_sequence]

        # ===== Step 2 - From relative leaf ranks to absolute leaf ranks (abnormal ranks removing) ====================

        if rank_attribution:

            abnormal_ranks = detect_abnormal_ranks(alignment_matrix)

            # update sequence attributes
            for snapshot in self.snapshots:
                snapshot.sequence = [e for i, e in enumerate(snapshot.sequence) if i not in abnormal_ranks]

    def growing_leaf_tracking(self):
        """
        Tracking of growing leaves over time.
        To use AFTER self.align_mature()
        """

        # avoid long time gaps in the time-series
        valid_snapshots = [snapshot for snapshot in self.snapshots if snapshot.check_continuity]

        mature_ref = self.get_ref_skeleton()

        for r in mature_ref.keys():

            # init leaf ref
            leaf_ref = mature_ref[r]

            # day t when leaf starts to be mature
            t_mature = next((t for t, snapshot in enumerate(valid_snapshots) if snapshot.sequence[r] != -1))

            # backwards tracking of this leaf
            for t in range(t_mature)[::-1]:
                snapshot = valid_snapshots[t]
                g_growing = [g for g, leaf in enumerate(snapshot.leaves)
                             if (leaf.features['mature'] == False)  # avoids non-tracked mature
                             and (g not in snapshot.sequence)  # avoids already-tracked growing
                             ]
                if len(g_growing) > 0:
                    dists = [leaf_polylines_distance(polyline_ref=leaf_ref.polyline,
                                                     polyline_candidate=snapshot.leaves[g].polyline)
                             for g in g_growing]
                    valid_snapshots[t].sequence[r] = g_growing[np.argmin(dists)]

    # def polylines_and_ranks(self, mature=True, growing=True):
    #     pass

    def output(self):

        ranks = [snapshot.leaf_ranks() for snapshot in self.snapshots]
        checks = np.array([snapshot.check_continuity for snapshot in self.snapshots])
        return ranks, checks
