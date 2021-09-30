import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from openalea.maizetrack.alignment import multi_alignment, detect_abnormal_ranks, phm_leaves_distance

from openalea.maizetrack.utils import simplify, rgb_and_polylines, get_rgb, missing_data
from openalea.maizetrack.phenomenal_display import plot_leaves
from openalea.maizetrack.stem_correction import abnormal_stem

from openalea.phenomenal.object.voxelSegmentation import VoxelSegmentation
from openalea.phenomenal.object.voxelOrgan import VoxelOrgan


class TrackedLeaf(VoxelOrgan):

    def __init__(self, leaf):
        super().__init__(label=leaf.label, sub_label=leaf.sub_label)
        self.voxel_segments = leaf.voxel_segments
        self.info = leaf.info

        self.highest_pl = ()
        self.real_pl = ()

        self.rank_annotation = None

        # TODO : exact value of h_stem ??
        # insertion height (starting from stem tip for growing leaves)
        #h_stem = -700
        # TODO : takes time..
        #self.height = self.real_longest_polyline()[0][2] - h_stem

        self.azimuth = leaf.info['pm_azimuth_angle']
        if leaf.info['pm_label'] == 'mature_leaf':
            self.length = leaf.info['pm_length']
            if 'pm_z_base_voxel' in leaf.info:
                self.height = leaf.info['pm_z_base_voxel']
            else:
                self.height = leaf.info['pm_z_base']
                print('no voxel height !')
        else:
            self.length = leaf.info['pm_length_with_speudo_stem']
            self.height = leaf.info['pm_z_base']

        self.vec = [self.height, self.length, self.azimuth]


class TrackedSnapshot(VoxelSegmentation):

    def __init__(self, vmsi, metainfo, order):
        super().__init__(voxels_size=vmsi.voxels_size)
        self.voxel_organs = vmsi.voxel_organs
        self.info = vmsi.info

        # reset leaves initial order in "leaves", using the phenonemal ordering method (topology)
        # from openalea.phenomenal.segmentation.maize_analysis import maize_growing_leaf_analysis_real_length
        # mature_leafs = [(l, l.info["pm_z_base"]) for l in self.get_mature_leafs()]
        # mature_leafs.sort(key=lambda x: x[1])
        # growing_leafs = [(l, maize_growing_leaf_analysis_real_length(self, l)) for l in self.get_growing_leafs()]
        # growing_leafs.sort(key=lambda x: x[1])
        # for leaf_number, (l, _) in enumerate(mature_leafs + growing_leafs):
        #     l.info["pm_leaf_number"] = leaf_number + 1

        # initial leaves ordering with phenomenal order (given in leaf.info['pm_leaf_number'])
        leaves = [self.get_leaf_order(k) for k in range(1, 1 + self.get_number_of_leaf())]
        self.leaves = [TrackedLeaf(leaf) for leaf in leaves]

        self.metainfo = metainfo
        self.order = order
        self.image = dict()
        self.rank_annotation = [-2] * len(self.leaves)

    def snapshot_ar(self):
        # return np.array([l.vec for l in self.leaves])
        s_vec = len(self.leaves[0].vec)
        return np.array([self.leaves[i].vec if i != '-' else [np.NAN] * s_vec for i in self.order])
        # TODO : used in align_mature, where i != '-' is not possible ?

    # def add_gap(self, i_gaps):
    #     for i in i_gaps:
    #         self.order = self.order[:i] + ['-'] + self.order[i:]

    def permute(self, order1, order2):
        self.order[order1], self.order[order2] = self.order[order2], self.order[order1]

    def get_ranks(self):
        return [self.order.index(i) if i in self.order else -1 for i, leaf in enumerate(self.leaves)]


class TrackedPlant:

    def __init__(self, snapshots, abnormal_vmsi_list, plantid):

        self.snapshots = snapshots
        self.abnormal_vmsi_list = abnormal_vmsi_list
        self.plantid = plantid
        self.var_names = ['height', 'length', 'azimuth']

        for vmsi in self.abnormal_vmsi_list:
            for leaf in vmsi.get_leafs():
                leaf.info['pm_leaf_number_tracking'] = 0

    # def standardization(self):
    #
    #     # height and length normalization
    #     heights = np.array([leaf.vec[0] for snapshot in self.snapshots for leaf in snapshot.leaves])
    #     lengths = np.array([leaf.vec[1] for snapshot in self.snapshots for leaf in snapshot.leaves])
    #     for snapshot in self.snapshots:
    #         for leaf in snapshot.leaves:
    #             leaf.vec[0] = (leaf.vec[0] - np.mean(heights)) / (np.std(heights))
    #             leaf.vec[1] = (leaf.vec[1] - np.mean(lengths)) / (np.std(lengths))
    #
    #     # azimuth [-180, 180] -> [-1, 1]
    #     for snapshot in self.snapshots:
    #         for leaf in snapshot.leaves:
    #             leaf.vec[2] /= 180

    # TODO : not here ?
    def compute_vectors(self, w_h):
        """ new method to compute x,y,z vectors """
        for snapshot in self.snapshots:
            for leaf in snapshot.leaves:
                a = leaf.azimuth / 360 * 2 * np.pi
                h = leaf.height
                leaf.vec = np.array([np.cos(a), np.sin(a), w_h * h])

    def simplify_polylines(self):

        for snapshot in self.snapshots:

            for leaf in snapshot.leaves:
                leaf.highest_pl = simplify(leaf.get_highest_polyline().polyline, 50)
                leaf.real_pl = simplify(leaf.real_longest_polyline(), 50)

    @staticmethod
    def load_and_check(vmsi_list, check_stem=True, discontinuity=5):
        """ Update 13/09 : takes list of vmsi as input, each vmsi having a metainfo attribute"""

        # TODO : arg json=None, ou on peut donner le json qui contient 3 infos : ref_time, ref_order, orders=M

        # sort vmsi objects by time
        timestamps = np.array([vmsi.metainfo.timestamp for vmsi in vmsi_list])
        order = sorted(range(len(timestamps)), key=lambda k: timestamps[k])
        vmsi_list = [vmsi_list[i] for i in order]

        plantid = int(vmsi_list[0].metainfo.plant[:4])

        # check abnormal vmsi objects
        # a - remove vmsi with missing data
        checks_data = [not missing_data(v) for v in vmsi_list]
        print('{} vmsi to remove because of missing data'.format(len(checks_data) - sum(checks_data)))
        # b - vmsi with stem shape problems
        if check_stem:
            checks_stem = [not b for b in abnormal_stem(vmsi_list)]
            print('{} vmsi to remove because of stem shape abnormality'.format(len(checks_stem) - sum(checks_stem)))
        else:
            checks_stem = [True] * len(vmsi_list)
            print('no stem shape abnormality checking')

        # check if there is no big time gap in the time-series
        timestamps2 = [vmsi.metainfo.timestamp for vmsi in vmsi_list] # vmsi_list is ordered this time
        checks_continuity = np.array([True] * len(vmsi_list))
        dt_mean = np.mean(np.diff(timestamps2))
        for i in range(1, len(timestamps2)):
            if (timestamps2[i] - timestamps2[i - 1]) > discontinuity * dt_mean:
                checks_continuity[i:] = False
        print('{} vmsi to remove because of time gap'.format(len(checks_continuity) - sum(checks_continuity)))

        checks = list((np.array(checks_data) * np.array(checks_stem) * np.array(checks_continuity)).astype(int))

        # init alignment matrix
        normal_vmsi_list = [vmsi for check, vmsi in zip(checks, vmsi_list) if check]
        abnormal_vmsi_list = [vmsi for check, vmsi in zip(checks, vmsi_list) if not check]
        orders = [list(range(v.get_number_of_leaf())) for v in normal_vmsi_list]
        max_len = max([len(order) for order in orders])
        orders = [order + [-1] * (max_len - len(order)) for order in orders]

        print('creating snapshots / leaves')
        snapshots = [TrackedSnapshot(vmsi, vmsi.metainfo, order) for vmsi, order in zip(normal_vmsi_list, orders)]
        print('ok')
        trackedplant = TrackedPlant(snapshots=snapshots, abnormal_vmsi_list=abnormal_vmsi_list, plantid=plantid)

        return trackedplant

    def load_images(self, angle):

        for snapshot in self.snapshots:
            snapshot.image[angle], _ = get_rgb(metainfo=snapshot.metainfo, angle=angle)

    # TODO : move in rank_annotation ?
    def load_rank_annotation(self):
        """
        Load rank annotation for each leaf of each snapshot (-2 = not annotated, -1 = anomaly, 0 = rank 1, 1 = rank 2,
        etc.) Each annotation is associated to the x,y,z tip position of its corresponding leaf, in a csv
        """

        # TODO : don't stock annotated ranks in both snapshot and leaf..

        df_path = 'rank_annotation/rank_annotation_{}.csv'.format(self.plantid)

        if os.path.isfile(df_path):
            df = pd.read_csv(df_path)

            for snapshot in self.snapshots:
                task = snapshot.metainfo.task

                if task not in df['task'].unique():
                    # a task was not annotated
                    snapshot.rank_annotation = [-2] * len(snapshot.leaves)

                    for leaf in snapshot.leaves:
                        leaf.rank_annotation = -2
                else:
                    dftask = df[df['task'] == task]
                    snapshot.rank_annotation = []
                    for leaf in snapshot.leaves:
                        tip = leaf.real_longest_polyline()[-1]
                        dftip = dftask[dftask['leaf_tip'] == str(tip)]
                        if dftip.empty:
                            # a leaf was not annotated
                            snapshot.rank_annotation.append(-2)
                            leaf.rank_annotation = -2
                        else:
                            snapshot.rank_annotation.append(dftip.iloc[0]['rank'])
                            leaf.rank_annotation = dftip.iloc[0]['rank']

    # TODO : move in rank_annotation ?
    def save_rank_annotation(self):

        df = []

        save = True

        for snapshot in self.snapshots:

            # verify that a rank was not asign severals times
            ranks = [r for r in snapshot.rank_annotation if r not in [-2, -1]]
            if len(ranks) != len(set(ranks)):
                print('task', snapshot.metainfo.task, ': several leaves have the same rank ! Cannot save')
                save = False
            else:
                for leaf, rank in zip(snapshot.leaves, snapshot.rank_annotation):
                    leaf_tip = str(leaf.real_longest_polyline()[-1])
                    df.append([snapshot.metainfo.task, leaf_tip, rank])

        df = pd.DataFrame(df, columns=['task', 'leaf_tip', 'rank'])

        if save:
            df.to_csv('rank_annotation/rank_annotation_{}.csv'.format(self.plantid), index=False)

    def get_dataframe(self, load_anot=True):
        """
        Summarize data for all snapshots that were tracked, in a dataframe
        """

        if load_anot:
            self.load_rank_annotation()

        df = []

        for snapshot in self.snapshots:

            annotation = np.array(snapshot.rank_annotation) + 1
            #pred = np.array(snapshot.get_ranks()) + 1

            for i, leaf in enumerate(snapshot.leaves):

                mature = leaf.info['pm_label'] == 'mature_leaf'
                h, l, a = (leaf.height, leaf.length, leaf.azimuth)
                df.append([self.plantid, snapshot.metainfo.timestamp, mature,
                                       leaf.info['pm_leaf_number'], leaf.info['pm_leaf_number_tracking'], annotation[i],
                                       h, l, a, leaf.info['pm_length_extended'], leaf.info['pm_z_base']])

        df = pd.DataFrame(df, columns=['plantid', 'timestamp', 'mature',
                                   'rank_phenomenal', 'rank_tracking', 'rank_annotation',
                                   'h', 'l', 'a', 'l_extended', 'h_old'])

        return df

    def evaluate_tracking(self):

        df = self.get_dataframe()
        a = len(df[(df['mature'] == True) & (df['obs'] == df['pred'])])
        b = len(df[df['mature'] == True])
        print('{}/{} mature leaves correclty classified'.format(a, b))
        c = len(df[(df['mature'] == False) & (df['obs'] == df['pred'])])
        d = len(df[df['mature'] == False])
        print('{}/{} growing leaves correclty classified'.format(c, d))

    def motif_ar(self, i_snapshots):
        # return a 3D array of shape (nb of snapshots, snapshot/motif length, leaf vec length)
        # all input snapshots need to have the same length !!

        snp_ar = [self.snapshots[i].snapshot_ar() for i in i_snapshots]

        if not all([len(k) == len(snp_ar[0]) for k in snp_ar]):
            print('motif_ar() : different lengths !')
            return

        return np.stack(snp_ar)

    # def update(self, i_snapshots, r):
    #
    #     new_gaps = [g for g in range(len(r)) if r[g] == '-']
    #     if new_gaps != []:
    #         for i in i_snapshots:
    #             self.snapshots[i].add_gap(new_gaps)

    # def alignment_matrix(self):
    #
    #     mat = [snapshot.order for snapshot in self.snapshots]
    #     mat = [[int(e != '-') for e in sp] for sp in mat]
    #     return mat

    def get_ref_skeleton(self, display=False):

        ref_skeleton = dict()

        ranks = range(len(self.snapshots[0].order))
        for rank in ranks:

            leaves = [s.leaves[s.order[rank]] for s in self.snapshots if s.order[rank] != -1]
            leaves = [leaf for leaf in leaves if leaf.info['pm_label'] == 'mature_leaf']

            # TODO : bricolage
            # remove old leaves (that could have a different shape)
            leaves = leaves[:15]
            # remove young leaves (that could have a different shape)
            if len(leaves) > 5:
                leaves = leaves[2:]

            vec_mean = np.mean(np.array([l.vec for l in leaves]), axis=0)
            dists = [np.sum(abs(l.vec - vec_mean)) for l in leaves]

            ref_skeleton[rank] = leaves[np.argmin(dists)]

        if display:
            plot_leaves([ref_skeleton[r] for r in ranks], ranks)

        return ref_skeleton

    def tracking_info_update(self):
        for snapshot in self.snapshots:
            ranks = snapshot.get_ranks()
            for leaf, rank in zip(snapshot.leaves, ranks):
                leaf.info['pm_leaf_number_tracking'] = rank + 1

    def align_mature(self, gap=1.5, gap_extremity_factor=0.5, direction=1, n_previous=5, w_h=0.001):

        # Step 1 - multi sequence alignment
        # ==============================================

        self.compute_vectors(w_h=w_h)

        # init order attribute of each snapshot, with only mature leaves:
        for snapshot in self.snapshots:
            snapshot.order = [i for i, l in enumerate(snapshot.leaves) if l.info['pm_label'] == 'mature_leaf']

        sequences = [snapshot.snapshot_ar() for snapshot in self.snapshots]

        alignment_matrix = multi_alignment(sequences, gap, gap_extremity_factor, direction, n_previous)

        # update order attributes
        for i, order in enumerate(alignment_matrix):
            self.snapshots[i].order = list(order)

        # Step 2 - abnormal ranks removing
        # ==============================================

        abnormal_ranks = detect_abnormal_ranks(alignment_matrix)

        # update order attributes
        for snapshot in self.snapshots:
            snapshot.order = [e for i, e in enumerate(snapshot.order) if i not in abnormal_ranks]

        print('{}/{} ranks removed'.format(len(abnormal_ranks), len(alignment_matrix[0])))

        # ============ saving results in leaf.info attributes

        # update leaf.info
        self.tracking_info_update()

        return alignment_matrix

    def align_growing(self):

        self.simplify_polylines()

        mature_ref = self.get_ref_skeleton(display=False)

        for r in mature_ref.keys():

            # init leaf ref
            leaf_ref = mature_ref[r]

            # day t when leaf starts to be mature
            t_mature = next((t for t, snapshot in enumerate(self.snapshots) if snapshot.order[r] != -1))

            for t in range(t_mature)[::-1]:

                snapshot = self.snapshots[t]
                g_growing = [g for g in range(len(snapshot.leaves)) if g not in snapshot.order]

                # plot_leaves([leaf_ref] + [snapshot.leaves[g] for g in g_growing], [-1] + [k for k in range(len(g_growing))])

                if g_growing != []:

                    g_min, d_min = -1, float('inf')
                    for g in g_growing:
                        leaf_candidate = snapshot.leaves[g]
                        d = phm_leaves_distance(leaf_ref=leaf_ref, leaf_candidate=leaf_candidate, method=2)
                        if d < d_min:
                            g_min, d_min = g, d

                    if g_min == -1:
                        print('problem : ', t, r, g_growing, d)

                    # print('rank = {}, t = {}, d = {}, (g = {})'.format(r, t, round(d_min, 2), g_min))
                    if self.snapshots[t].order[r] != -1:
                        print(' !! ')
                    self.snapshots[t].order[r] = g_min

        # update leaf.info
        self.tracking_info_update()

    def display(self, dates=None, stem=False):

        if dates is None:
            snapshots = self.snapshots
        else:
            snapshots = [snapshot for snapshot in self.snapshots if snapshot.metainfo.daydate in dates]

        leaves, ranks = [], []
        for snapshot in snapshots:
            leaves += [snapshot.leaves[i_leaf] for i_leaf in snapshot.order if i_leaf != -1]
            ranks += [i for i, i_leaf in enumerate(snapshot.order) if i_leaf != -1]

        plot_leaves(leaves, ranks)

    def save_gif(self, angle, path, fps=1):

        self.load_images(angle)

        imgs = []
        for snapshot in self.snapshots:

            ranks = snapshot.get_ranks()
            img, _ = rgb_and_polylines(snapshot, angle=angle, ranks=ranks)
            imgs.append(img)

        imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs]
        imgs_gif[0].save(path,
                         save_all=True,
                         append_images=imgs_gif[1:],
                         optimize=True,
                         duration=1000 / fps,
                         loop=0)

    def dump(self):

        res = {}
        for snapshot in self.snapshots:
            # ranks = snapshot.get_ranks()
            # for leaf, rank in zip(snapshot.leaves, ranks):
            #     leaf.info['pm_leaf_number_tracking'] = rank + 1

            # snapshot to vmsi conversion
            vmsi = VoxelSegmentation(voxels_size=snapshot.voxels_size)
            vmsi.voxel_organs = snapshot.voxel_organs
            vmsi.info = snapshot.info

            res[snapshot.metainfo.timestamp] = vmsi

        for vmsi in self.abnormal_vmsi_list:
            # for leaf in vmsi.get_leafs():
            #     leaf.info['pm_leaf_number_tracking'] = 0

            res[vmsi.metainfo.timestamp] = vmsi

        return res


def show_alignment_score(trackedplant, dt=3, only_score=False):

    # matrix containing all plant vecs (empty or not)
    mat = trackedplant.motif_ar(np.arange(0, len(trackedplant.snapshots)))

    mat = mat.transpose(1, 0, 2)[::-1]

    R, T, nvar = mat.shape
    mat_score = np.zeros((R, T)) * np.NAN
    for r in range(R):
        for t in range(T):
            vec = mat[r, t]
            if not all(np.isnan(vec)):

                # surrounding vecs between t - dt and t + dt
                surrounding_vecs = mat[r, max(0, t - dt):(t + dt + 1)]
                # remove vec in surrounding_vecs
                surrounding_vecs = np.array([v for v in surrounding_vecs if not np.all(v == vec)])
                # remove empty vec
                surrounding_vecs = surrounding_vecs[[not all(np.isnan(l)) for l in surrounding_vecs]]

                if len(surrounding_vecs) == 0:
                    score = 0
                else:
                    scores = [substitution_function(vec, veci, 0) for veci in surrounding_vecs]
                    score = np.mean(scores)

                #mat_score[r, t] = score
                mat_score[r, t] = score


    if only_score:
        # heat map for score
        plt.figure()
        plt.imshow(mat_score, cmap='summer', vmin=0, vmax=3)
        plt.xticks(np.arange(T), [trackedplant.snapshots[t].metainfo.daydate[5:] for t in range(T)], rotation='vertical')
        plt.yticks(np.arange(R), R - np.arange(R))
        plt.ylabel('Leaf rank')
        plt.xlabel('Day')
        plt.title('Similarity with other leaves of same rank (plantid {})'.format(trackedplant.plantid))

        im_ratio = mat_score.shape[0] / mat_score.shape[1]
        plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)

    else:

        # TODO: broken

        fig, axes = plt.subplots(nvar + 1, 1)
        fig.suptitle('plantid' + str(trackedplant.plantid))

        # 1 heat map / variable
        for k in range(nvar):
            plt.sca(axes[k])
            mat_var = mat[:, :, k]
            mat_var[pd.isnull(mat_var)] = -5
            mat_var = mat_var.astype('float64')
            plt.imshow(mat_var, cmap='hot')
            plt.xticks(np.arange(T), [trackedplant.snapshots[t].metainfo.daydate[5:] for t in range(T)], rotation='vertical', fontsize=5)
            plt.yticks(np.arange(R), R - np.arange(R), fontsize=8)
            plt.ylabel('Rank')
            plt.xlabel('Day')

            plt.title(trackedplant.var_names[k])

        # heat map for score
        plt.sca(axes[nvar])
        plt.imshow(mat_score)
        plt.xticks(np.arange(T), [trackedplant.snapshots[t].metainfo.daydate[5:] for t in range(T)], rotation='vertical', fontsize=5)
        plt.yticks(np.arange(R), R - np.arange(R), fontsize=8)
        plt.ylabel('Rank')
        plt.xlabel('Day')
        plt.title('Score')




#######################################################################
# Deprecated functions
#######################################################################


def show_alignment_score2(plant, ranks):

    # TODO : doesn't work, need to be corrected

    leaves = []
    ranks_obs = []
    ranks_pred = []
    times = []
    for t, snapshot in enumerate(plant.snapshots):
        leaves += snapshot.leaves
        ranks_pred += snapshot.get_ranks()
        ranks_obs += snapshot.rank_annotation
        times += [t] * len(snapshot.leaves)

    R, T = max(ranks) + 1, max(times) + 1
    mat = np.full((R, T), None)
    mat_anot = np.full((R, T), True)
    for robs, rpred, t, leaf in zip(ranks_obs, ranks_pred, times, leaves):

        if rpred != -1:
            mat[rpred, t] = leaf

            if robs != rpred:
                mat_anot[rpred, t] = False
                print(rpred, t)

    mature_ranks = [[r for r, l in zip(s.get_ranks(), s.leaves) if l.info['pm_label'] == 'mature_leaf'] for s in
                    plant.snapshots]
    max_mature_rank = [1 + max(ranks) if ranks != [] else 0 for ranks in mature_ranks]

    limit = []
    xx, yy = [], []
    r_previous = -1
    for t, r in enumerate(max_mature_rank):
        if r != r_previous:
            limit.append([t, r_previous])
            limit.append([t, r])
        r_previous = r

    R, T = mat.shape
    mat_plt_growing = np.zeros((R, T)) * np.NAN
    mat_plt_mature = np.zeros((R, T)) * np.NAN
    for r in range(R):
        for t in range(T):

            leaf = mat[r, t]
            if leaf is not None:

                if leaf.info['pm_label'] == 'mature_leaf':

                    mat_plt_mature[r, t] = 1

                elif leaf.info['pm_label'] == 'growing_leaf':

                    if t < T:
                        # mat_plt[r, t] = 3 * np.random.random()
                        previous_leaves = [mat[r, ti] for ti in range(t + 1, T) if mat[r, ti] is not None]
                        next_leaves = [mat[r, ti] for ti in range(0, t) if mat[r, ti] is not None]
                        if previous_leaves != []:
                            d1 = phm_leaves_distance(leaf, previous_leaves[-1], method=2)
                            pl1 = leaf.real_pl
                            leaf_len = np.sum([np.linalg.norm(np.array(pl1[k]) - np.array(pl1[k + 1])) for k in
                                               range(len(pl1) - 1)])
                            xx.append(leaf_len)
                            yy.append(d1)

                        else:
                            d1 = 0
                        if next_leaves != []:
                            d2 = phm_leaves_distance(leaf, next_leaves[0], method=2)
                        else:
                            d2 = 0
                        if d1 == 0:
                            d = d2
                        elif d2 == 0:
                            d = d1
                        else:
                            d = (d1 + d2) / 2
                        mat_plt_growing[r, t] = d

    plt.figure()
    # vmax = np.nanmax(mat_plt)
    vmax = 30
    plt.imshow(mat_plt_growing[::-1], cmap='summer', vmin=0, vmax=vmax)
    plt.imshow(mat_plt_mature[::-1], cmap='hot')
    # plt.plot(np.array(limit)[:, 0] - 0.5, np.array(limit)[:, 1] - 0.5, 'r-')
    plt.ylim(None, -0.5)

    for r in range(R):
        for t in range(T):
            if mat_anot[::-1][r, t] == False:
                plt.plot(t, r, 'r*', markersize=3)


def permut_mature(plant, gap):

    # TODO : doesnt work

    dt = 7
    permutation_penalty = 0.7
    direction = 1
    for i in range(len(plant.snapshots))[::direction]:

        xi = np.concatenate(
            (np.arange(max(i - dt, 0), i), np.arange(i + 1, min(i + 1 + dt, len(plant.snapshots)))))
        yi = [i]

        X = plant.motif_ar(xi)  # ref
        Y = plant.motif_ar(yi)  # to try permutations

        ref_score = nw_score(X, Y, gap)

        i_leaves = [i for i, l in enumerate(Y[0]) if not all(np.isnan(l))]

        if len(i_leaves) > 1:  # can not permute if only 1 leaf in Y
            for k in range(len(i_leaves) - 1):
                Y2 = Y.copy()
                Y2[0][[i_leaves[k], i_leaves[k + 1]]] = Y2[0][[i_leaves[k + 1], i_leaves[k]]]
                score = nw_score(X, Y2, gap)
                if score < ref_score:
                    print('{} : ranks {} & {}. Score gain : {}. Enough to permute : {}'.format(
                        plant.snapshots[i].metainfo.daydate,
                        i_leaves[k],
                        i_leaves[k + 1],
                        round(ref_score - score, 1),
                        (score + permutation_penalty < ref_score)
                        ))

                    if score + permutation_penalty < ref_score:
                        # i_leaves[k] = position in sq.order of leaf
                        plant.snapshots[i].permute(i_leaves[k], i_leaves[k + 1])


def deprecated_align_growing(plant):

    # TODO : doesn't work

    mature_ref, _ = plant.get_ref_skeleton()

    for i_seq in range(len(plant.snapshots))[::-1]:

        # the 2 sequences to compare
        seq_ref = plant.snapshots[i_seq]
        seq_next = plant.snapshots[i_seq - 1]

        # plantid of growing leaves than can be ordered.
        g_growing = [g for g, l in enumerate(seq_next.leaves) if l.info['pm_label'] == 'growing_leaf']

        # where theses leaves can be put
        i_orders = []
        for i in range(len(seq_ref.order)):
            if seq_ref.order[i] != '-' and seq_next.order[i] == '-':
                if plant.alignment_reference()[i] != -1:  # to not add leaves to a "gap rank"
                    i_orders.append(i)

        # TODO bricolage !
        for i_order in i_orders:
            if i_order + 5 < max(i_orders):
                i_orders.remove(i_order)

        plot_ = False
        if plot_:
            Lg = [seq_next.leaves[g] for g in g_growing]
            Ls = [seq_ref.leaves[seq_ref.order[i_order]] for i_order in i_orders]
            # plot_leaves(Lg + Ls, [-1] * len(Lg) + [plant.alignment_reference()[i_order] for i_order in i_orders])

        print('starting')
        for i_order in i_orders:

            leaf_ref = seq_ref.leaves[seq_ref.order[i_order]]
            # si mature, tant qu'a faire autant selectionner la "meilleure" = celle du skeleton de ref
            if leaf_ref.info['pm_label'] == 'mature_leaf':
                leaf_ref = mature_ref[i_order]

            g_min, d_min = -1, 5000000

            if g_growing != []:  # if there are still some leaves to order

                for g in g_growing:
                    leaf_next = seq_next.leaves[g]
                    d = phm_leaves_distance(leaf_ref, leaf_next)
                    if d < d_min:
                        g_min, d_min = g, d

                print('New leaf (plantid {}) in snapshot {} at rank {} (dist = {})'.format(g_min,
                                                                                           seq_next.metainfo.daydate,
                                                                                           i_order,
                                                                                           round(d_min, 1)))
                plant.snapshots[i_seq - 1].order[i_order] = g_min

                # remove ordered leaf from non-ordered list
                g_growing.remove(g_min)

        # tempo
        print('done !')
        i_seq -= 1
        print(i_seq, i_seq - 1)


def iterative_mature(plant, gap):

    # TODO : doesn't work

    # ITERATIVE
    dt = 5
    direction = 1
    print('iterative, direction = ', direction)
    for i in range(len(plant.snapshots))[::direction]:
        xi = np.concatenate((np.arange(max(i - dt, 0), i), np.arange(i + 1, min(i + 1 + dt, len(plant.snapshots)))))
        yi = [i]

        X = plant.motif_ar(xi)  # ref
        Y = plant.motif_ar(yi)  # to try permutations
        Y = np.array([[list(y) for y in Y[0] if not all(np.isnan(y))]])  # remove gaps

        rx, ry = needleman_wunsch(X, Y, gap, gap_extremity=gap * 1)

        # TODO : and if rx changes ???????

        if ry != plant.snapshots[i].order:
            # TODO : a revoir, a coder plus proprement
            i_leaves = np.array([k for k in plant.snapshots[i].order if k != '-'])
            i_ry = np.array([i for i in range(len(ry)) if ry[i] != '-'])
            for ii in range(len(i_ry)):
                ry[i_ry[ii]] = i_leaves[ii]
            if ry != plant.snapshots[i].order:
                plant.snapshots[i].order = ry
                print('Update', plant.snapshots[i].metainfo.daydate)


