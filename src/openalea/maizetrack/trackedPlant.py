"""
Classes for maize leaf tracking and rank attribution, in a time-series of Phenomenal 3D segmentation objects.
//!\\ In the tracking algorithm, ranks start at 0. But in the final output (stored in the 'pm_leaf_number_tracking'
attribute of each leaf), ranks start at 1. (see get_ranks() method)
"""

import os
import numpy as np
import pandas as pd
import warnings

from openalea.maizetrack.alignment import multi_alignment, detect_abnormal_ranks, phm_leaves_distance
from openalea.maizetrack.utils import simplify, get_rgb, missing_data
from openalea.maizetrack.stem_correction import abnormal_stem
from openalea.maizetrack.phenomenal_display import has_pgl_display, plot_leaves

from openalea.phenomenal.object.voxelSegmentation import VoxelSegmentation
from openalea.phenomenal.object.voxelOrgan import VoxelOrgan


class TrackedLeaf(VoxelOrgan):
    """
    class inherited from openalea.phenomenal.object.voxelOrgan.VoxelOrgan
    Describe a leaf organ, with attributes specific to leaf tracking algorithm.
    """

    def __init__(self, leaf):
        super().__init__(label=leaf.label, sub_label=leaf.sub_label)
        self.voxel_segments = leaf.voxel_segments
        self.info = leaf.info

        self.highest_pl = ()
        self.real_pl = ()

        self.check = {'no_senescence': True, 'no_alignment_anomaly': True}

        self.features = {}
        self.vec = np.array([])

        self.rank_annotation = None


class TrackedSnapshot(VoxelSegmentation):
    """
    class inherited from openalea.phenomenal.object.voxelSegmentation.VoxelSegmentation
    Describe the plant segmentation from Phenomenal at a given time point, and associate it to its corresponding
    metainfos. Describe the order of leaves, which is modified during leaf tracking.
    """

    def __init__(self, vmsi, metainfo, check):
        super().__init__(voxels_size=vmsi.voxels_size)
        self.voxel_organs = vmsi.voxel_organs
        self.info = vmsi.info

        # list of TrackedLeaf objects
        # (ordered in a sequence using phenomenal topological order, given in leaf.info['pm_leaf_number'])
        leaves = [self.get_leaf_order(k) for k in range(1, 1 + self.get_number_of_leaf())]
        self.leaves = [TrackedLeaf(leaf) for leaf in leaves]

        self.check = check

        # self.sequence gives the ranks of leaves in self.leaves.
        # for example, if self.order[5] = 2, it means that self.leaves[2] is associated to rank 5+1=6.
        # -1 = no leaf
        self.sequence = []

        self.metainfo = metainfo

        # not useful for the tracking algorithm itself, only used for ground-truth annotation, display, etc.
        self.image = dict()
        self.rank_annotation = [-2] * len(self.leaves)

    def get_ranks(self):
        """
        list of ranks of leaves contained in self.leaves

        small example :
        self.leaves = [leaf0, leaf1, leaf2, leaf3]
        self.sequence = [-1, -1, 0, 1, -1, 2, 3, -1]
        ===> self.get_ranks() returns [3, 4, 6, 7]

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

        # # abnormals vmsi from TrackedPlant.load_and_check()
        # self.abnormal_vmsi_list = abnormal_vmsi_list

        # self.plantid = plantid
        #
        # # init : no tracking yet, all leaf ranks equal to 0
        # for vmsi in self.abnormal_vmsi_list:
        #     for leaf in vmsi.get_leafs():
        #         leaf.info['pm_leaf_number_tracking'] = 0

    # ===== method to load a TrackedPlant object ====================================================================

    @staticmethod
    def load_and_check(vmsi_list, discontinuity=5.):
        """
        Load a TrackedPlant object from a list of vmsi + metainfos.
        Check and set aside abnormal time points. Abnormal vmsi are stored in the 'abnormal_vmsi_list' attribute
        of the TrackedPlant object created. Normal vmsi are converted in TrackedSnapshot objects and stored in the
        'snapshots' attribute of the TrackedPlant object.

        Parameters
        ----------
        vmsi_list : list of openalea.phenomenal.object.voxelSegmentation.VoxelSegmentation (vmsi) objects
            Each vmsi must have a 'metainfo' attributes
        discontinuity : float
            parameter to remove big time gaps in the time-series of segmented objects

        Returns
        -------
        TrackedPlant object

        """

        # plantid = int(vmsi_list[0].metainfo.pot)

        # verify temporal order of the time-series
        timestamps = [v.metainfo.timestamp for v in vmsi_list]
        if timestamps != sorted(timestamps):
            raise Exception('objects need to be ordered by temporal order')

        # ===== anomaly detection =========================================================================

        # check if there is no big time gap in the time-series
        dt_median = np.median(np.diff(timestamps))
        checks_continuity = np.array([True] * len(vmsi_list))
        for i in range(1, len(timestamps)):
            if (timestamps[i] - timestamps[i - 1]) > discontinuity * dt_median:
                checks_continuity[i:] = False
        # print('{} time points with a time gap'.format(len(checks_continuity) - sum(checks_continuity)))

        # stem shape problems
        checks_stem = [not b for b in abnormal_stem(vmsi_list)]
        # print('{} time points with stem shape abnormality'.format(len(checks_stem) - sum(checks_stem)))

        # missing data (specific to Phenomenal)
        # TODO: seems to only affect growing leaf: len(leaf.info) = 7 instead of 23.
        checks_data = [not missing_data(v) for v in vmsi_list]
        # print('{} time points with missing data'.format(len(checks_data) - sum(checks_data)))

        # checks = list((np.array(checks_data) * np.array(checks_stem) * np.array(checks_continuity)).astype(int))

        # ===== create the TrackedPlant object =================================================================

        snapshots = []
        for k, vmsi in enumerate(vmsi_list):
            check = {'time_continuity': checks_continuity[k],
                     'valid_stem': checks_stem[k],
                     'valid_features': checks_data[k]}
            snapshots.append(TrackedSnapshot(vmsi, vmsi.metainfo, check))

        # # init alignment matrix
        # # normal_vmsi_list = [vmsi for check, vmsi in zip(checks, vmsi_list) if check]
        # # abnormal_vmsi_list = [vmsi for check, vmsi in zip(checks, vmsi_list) if not check]
        # orders = [list(range(v.get_number_of_leaf())) for v in normal_vmsi_list]
        # max_len = max([len(order) for order in orders])
        # orders = [order + [-1] * (max_len - len(order)) for order in orders]

        trackedplant = TrackedPlant(snapshots=snapshots)

        return trackedplant

    # ===========================================================================================================

    def valid_snapshots(self, check_list=None):

        checks = list(self.snapshots[0].check.keys()) if check_list is None else check_list
        return [s for s in self.snapshots if all([s.check[check] for check in checks])]

    # ===== Main methods used for the tracking algorithm ========================================================

    def features_extraction(self, w_h, w_l):
        """ computes a vector for each leaf of each snapshot. Used for exporting these data in a csv (a) and for
        the alignment of mature leaves (b) """

        # print('{} snapshots not used in features_extraction()'.format(len(self.snapshots) - len(snapshots)))
        for snapshot in self.snapshots:
            for leaf in snapshot.leaves:

                # ===== mature leaves ===========================================================================

                if leaf.info['pm_label'] == 'mature_leaf':

                    # a) complete 'features' attribute for exporting traits in csv
                    leaf.features['azimuth'] = leaf.info['pm_azimuth_angle']
                    leaf.features['length'] = leaf.info['pm_length']
                    if 'pm_z_base_voxel' in leaf.info:
                        leaf.features['height'] = leaf.info['pm_z_base_voxel']
                    else:
                        leaf.features['height'] = leaf.info['pm_z_base']
                        print('no voxel height !')

                    # b) compute a vector from these features for mature leaves alignment
                    a = leaf.features['azimuth'] / 360 * 2 * np.pi  # [0, 360] interval --> [-1, 1] interval
                    leaf.vec = np.array([np.cos(a), np.sin(a), w_h * leaf.features['height'],
                                         w_l * leaf.features['length']])

                # ===== growing leaves ===========================================================================

                elif leaf.info['pm_label'] == 'growing_leaf':

                    # a) complete 'features' attribute for exporting traits in csv.
                    # sometimes, they are missing data due to a bug in phenomenal: needs to be checked.
                    vars = ['pm_azimuth_angle', 'pm_length_with_speudo_stem', 'pm_z_base']
                    if not all([var in leaf.info for var in vars]):
                        leaf.features = {'azimuth': None, 'length': None, 'height': None}
                    else:
                        leaf.features = {'azimuth': leaf.info['pm_azimuth_angle'],
                                         'length': leaf.info['pm_length_with_speudo_stem'],
                                         'height': leaf.info['pm_z_base']}

    def simplify_polylines(self, new_length=50):
        """ simplify the polyline of each leaf in each snapshot (max number of points = 'new_length'), for a faster
         computation of growing leaves tracking """
        for snapshot in self.snapshots:
            for leaf in snapshot.leaves:
                # polyline starting from insertion point
                leaf.highest_pl = simplify(leaf.get_highest_polyline().polyline, new_length)
                # polyline starting from stem base
                leaf.real_pl = simplify(leaf.real_longest_polyline(), new_length)

    def get_ref_skeleton(self, display=False, nmax=15):
        """
        Compute a median skeleton {rank : leaf}.
        For each rank, the leaf whose vector is less distant to all other leaves from the same ranks is selected.

        Parameters
        ----------
        display : bool
            optionnal 3D display (True or False)
        nmax : int
            max number of leaves considered at a given rank (to avoid old leaves which can have senescence)

        Returns
        -------

        """

        ref_skeleton = dict()

        # TODO: no need to avoid "missing_features" check ?
        snapshots = self.valid_snapshots()

        ranks = range(len(snapshots[0].sequence))
        for rank in ranks:
            # all matures leaves for this rank
            leaves = [s.leaves[s.sequence[rank]] for s in snapshots if s.sequence[rank] != -1]
            leaves = [leaf for leaf in leaves if leaf.info['pm_label'] == 'mature_leaf']

            # remove old leaves (that could have a different shape)
            leaves = leaves[:nmax]

            if leaves:
                vec_mean = np.mean(np.array([l.vec for l in leaves]), axis=0)
                dists = [np.sum(abs(l.vec - vec_mean)) for l in leaves]
                ref_skeleton[rank] = leaves[np.argmin(dists)]

        if display:
            if has_pgl_display:
                plot_leaves(list(ref_skeleton.values()), list(ref_skeleton.keys()))
            else:
                warnings.warn('PlantGL not available')

        return ref_skeleton

    # def tracking_info_update(self):
    #     """
    #     update the 'info' attribute for each leaf of each snapshot.
    #     In the TrackedPlant object, rank data is saved and modified in the 'sequence' attribute of each snapshot, during
    #     the different steps of the tracking algorithms. Here, this functions allows to save the rank info for each leaf
    #     at the same place as other leaves info from phenomenal (leaf length, etc.)
    #     """
    #     for snapshot in self.snapshots:
    #         ranks = snapshot.get_ranks()
    #         for leaf, rank in zip(snapshot.leaves, ranks):
    #             leaf.info['pm_leaf_number_tracking'] = rank + 1

    def align_mature(self, gap=12.35, gap_extremity_factor=0.2, start=1, n_previous=5000, w_h=0.03, w_l=0.004,
                     rank_attribution=True, remove_senescence=False):
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

        self.features_extraction(w_h=w_h, w_l=w_l)

        # only check 'valid_stem'. 'time_continuity' and valid_features' are not requested for mature leaves.
        snapshots = self.valid_snapshots(check_list=['valid_stem'])

        # init sequence attribute of each snapshot, with only mature leaves:
        for snapshot in snapshots:
            # # TODO maj 15/06/2022, only for ZA17
            # if remove_senescence:
            #     zbase = {'elcom_2_c1_wide': -730, 'elcom_2_c2_wide': -690}
            #     snapshot.sequence = [i for i, l in enumerate(snapshot.leaves) if l.info['pm_label'] == 'mature_leaf' and
            #                       l.real_longest_polyline()[-1][2] - zbase[snapshot.metainfo.shooting_frame] > 10]
            # else:
            snapshot.sequence = [i for i, l in enumerate(snapshot.leaves) if l.info['pm_label'] == 'mature_leaf']

        # time-series of sequences of vectors (sequences have different lengths, all vectors have the same size)
        sequences = [np.array([snapshot.leaves[i].vec for i in snapshot.sequence]) for snapshot in snapshots]

        # sequence alignment
        alignment_matrix = multi_alignment(sequences, gap, gap_extremity_factor, n_previous, start)

        # update sequence attributes
        for i, sequence in enumerate(alignment_matrix):
            # self.snapshots[i].sequence = list(sequence)
            # TODO maj 15/06/2022 /!\
            snapshots[i].sequence = [k if k == -1 else snapshots[i].sequence[k] for k in sequence]

        # ===== Step 2 - From relative leaf ranks to absolute leaf ranks (abnormal ranks removing) ====================

        if rank_attribution:

            abnormal_ranks = detect_abnormal_ranks(alignment_matrix)

            for snapshot in snapshots:
                # save the information that some leaves are removed from the alignment
                for a in abnormal_ranks:
                    if snapshot.sequence[a] != -1:
                        snapshot.leaves[snapshot.sequence[a]].check['no_alignment_anomaly'] = False
                # update sequence attributes
                snapshot.sequence = [e for i, e in enumerate(snapshot.sequence) if i not in abnormal_ranks]

            print('{}/{} ranks removed after sequence alignment'.format(len(abnormal_ranks), len(alignment_matrix[0])))

        # # saving results in leaf.info attributes
        # # ================================================
        # self.tracking_info_update()

    def align_growing(self):
        """
        Tracking of growing leaves over time.
        To use AFTER self.align_mature()
        """

        self.simplify_polylines()

        # TODO: need for "missing_features" check, so different as for align_mature /!\
        snapshots = self.valid_snapshots()

        mature_ref = self.get_ref_skeleton(display=False)

        for r in mature_ref.keys():

            # init leaf ref
            leaf_ref = mature_ref[r]

            # day t when leaf starts to be mature
            t_mature = next((t for t, snapshot in enumerate(snapshots) if snapshot.sequence[r] != -1))

            for t in range(t_mature)[::-1]:

                snapshot = snapshots[t]
                g_growing = [g for g, leaf in enumerate(snapshot.leaves) if g not in snapshot.sequence]

                if g_growing != []:

                    g_min, d_min = -1, float('inf')
                    for g in g_growing:
                        leaf_candidate = snapshot.leaves[g]
                        d = phm_leaves_distance(leaf_ref=leaf_ref, leaf_candidate=leaf_candidate)
                        if d < d_min:
                            g_min, d_min = g, d

                    snapshots[t].sequence[r] = g_min

        # # update leaf.info
        # self.tracking_info_update()

    # ===== methods to output the full result of the tracking ==================================================

    # def dump(self):
    #     """ convert TrackedPlant object in a list of vmsi (normals and abnormals) After tracking, each leaf of each vmsi
    #     has its rank stored it its 'info' attribute """
    #
    #     res = {}
    #     for snapshot in self.snapshots:
    #         # snapshot to vmsi conversion
    #         vmsi = VoxelSegmentation(voxels_size=snapshot.voxels_size)
    #         vmsi.voxel_organs = snapshot.voxel_organs
    #         vmsi.info = snapshot.info
    #
    #         res[snapshot.metainfo.timestamp] = vmsi
    #
    #     for vmsi in self.abnormal_vmsi_list:
    #         res[vmsi.metainfo.timestamp] = vmsi
    #
    #     return res

    def get_dataframe(self, load_anot=False):
        """
        Summarize data for all snapshots that were tracked, in a dataframe
        """

        # if a ground-truth annotation is found, it is saved in the dataframe
        # TODO deprecated. (plantid not used anymore). Use anot path or anot csv as input of the function ?
        if load_anot:
            self.load_rank_annotation()

        df = []

        for snapshot in self.snapshots:

            # TODO deprecated: to remove
            annotation = np.array(snapshot.rank_annotation)

            ranks = np.array(snapshot.get_ranks())

            for i, leaf in enumerate(snapshot.leaves):
                mature = leaf.info['pm_label'] == 'mature_leaf'
                h, l, a = [leaf.features[k] for k in ['height', 'length', 'azimuth']] \
                    if snapshot.check['valid_features'] else 3 * [None]
                check_values = list(snapshot.check.values()) + list(leaf.check.values())
                check_columns = ['task_' + c for c in snapshot.check.keys()] + ['leaf_' + c for c in leaf.check.keys()]
                df.append([int(snapshot.metainfo.task), snapshot.metainfo.timestamp, mature,
                           leaf.info['pm_leaf_number'], ranks[i], annotation[i], h, l, a,
                           leaf.info['pm_length_extended']] + check_values)

        # l = normal phenomenal length. l_extended = length after leaf extension
        df = pd.DataFrame(df, columns=['task', 'timestamp', 'mature',
                                       'rank_phenomenal', 'rank_tracking', 'rank_annotation',
                                       'h', 'l', 'a', 'l_extended'] + check_columns)

        return df

    # ===== Methods for visualization and management of ground-truth annotations ==================================

    # def load_images(self, angle):
    #
    #     for snapshot in self.snapshots:
    #         snapshot.image[angle], _ = get_rgb(metainfo=snapshot.metainfo, angle=angle)

    # def display(self, dates=None, only_mature=False):
    #     """ 3D display using PlantGL (color=rank) """
    #
    #     if dates is None:
    #         snapshots = self.snapshots
    #     else:
    #         snapshots = [snapshot for snapshot in self.snapshots if snapshot.metainfo.daydate in dates]
    #
    #     leaves, ranks = [], []
    #     for snapshot in snapshots:
    #         leaves += [snapshot.leaves[i_leaf] for i_leaf in snapshot.sequence if i_leaf != -1]
    #         ranks += [i for i, i_leaf in enumerate(snapshot.sequence) if i_leaf != -1]
    #
    #     if only_mature:
    #         i_mature = [i for i, leaf in enumerate(leaves) if leaf.info['pm_label'] == 'mature_leaf']
    #         leaves = [leaves[i] for i in i_mature]
    #         ranks = [ranks[i] for i in i_mature]
    #
    #     if has_pgl_display:
    #         plot_leaves(leaves, ranks)
    #     else:
    #         warnings.warn('PlantGL not available')
