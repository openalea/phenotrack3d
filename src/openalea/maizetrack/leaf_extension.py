"""
Use binary images to extend the length of each phenomenal leaf.

Method : A 2d skeleton is computed for the binary image at a given angle. Then, the algorithm searches correspondences
between phenomenal polylines (reprojected in 2D) and skeleton polylines. For each match, an extension factor e >= 1 is
computed this way : e = (skeleton 2D polyline length) / (phenomenal 2D polyline length). This is done for each side
angle. Then, results are merged : for each phenomenal leaf, the final extension factor is equal to the mean of all
extension values found for this leaf, or 1 if no extension value was found.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skan import skeleton_to_csgraph, Skeleton, summarize
from scipy.spatial.distance import directed_hausdorff

import warnings

from openalea.maizetrack.utils import phm3d_to_px2d


def polyline_length(pl):
    # TODO : this function probably already exists in Phenomenal..?
    """

    Args:
        pl: polyline

    Returns: length of this polyline, computed the same way as in Phenomenal

    """
    return np.sum([np.linalg.norm(np.array(pl[k]) - np.array(pl[k + 1])) for k in range(len(pl) - 1)])


def skeleton_branches(img, n_kernel=15, min_length=30):
    """

    Args:
        img: binary image
        n_kernel: parameter for image preprocessing (dilating)
        min_length: minimum length of skeleton branches

    Returns: list of 2D polylines

    """

    # dilate image
    # TODO : try another method
    kernel = np.ones((n_kernel, n_kernel))
    img2 = cv2.dilate(img, kernel, iterations=1)

    # 2d skeleton image

    skeleton = skeletonize(img2)

    # skeleton analysis : get branches
    skan_skeleton = Skeleton(skeleton)
    branches = summarize(skan_skeleton)

    # select branches having an endpoint, and a sufficient length

    branches_endpoint = branches[branches['branch-type'] == 1]
    branches_endpoint = branches_endpoint[branches_endpoint['branch-distance'] > min_length]

    # converting branches to polylines

    _, coordinates, _ = skeleton_to_csgraph(skeleton)
    node_ids = list(branches['node-id-src']) + list(branches['node-id-dst'])
    polylines = []

    for irow, row in branches_endpoint.iterrows():

        polyline = np.array([coordinates[i] for i in skan_skeleton.path(irow)])
        polyline = polyline[:, ::-1]  # same (x, y) order as phenomenal

        # verify that all leaf polylines are oriented the same way (leaf insertion --> leaf tip)
        i = row['node-id-dst']
        if node_ids.count(i) > 1:
            polyline = polyline[::-1]

        polylines.append(polyline)

    return polylines


def compute_extension(polylines_phm, polylines_sk, seg_length=50., dist_threshold=30.):
    """

    Args:
        polylines_phm: list of phenomenal leaf polylines, projected in 2D
        polylines_sk: list of 2D skeleton polylines
        seg_length: length (px) of the end segment of a phenomenal leaf polyline that is compared with skeleton
        dist_threshold: minimum hausdorff distance (px) between phenomenal and skeleton polylines to associate them

    Returns:
    """

    res = dict.fromkeys(range(len(polylines_phm)), [])

    for pl_sk in polylines_sk:

        b_selected = 0
        dist_min = float('inf')
        selected_rank = -1

        for rank, pl_phm in enumerate(polylines_phm):

            # end segment of phenomenal polyline
            dists_to_end = np.linalg.norm(pl_phm - pl_phm[-1], axis=1)
            start = np.argmin(abs(dists_to_end - seg_length))
            pl_phm_segment = pl_phm[start:]

            # corresponding sk segment
            d_a = np.linalg.norm(pl_sk - pl_phm_segment[0], axis=1)
            a, min_a = np.argmin(d_a), np.min(d_a)
            d_b = np.linalg.norm(pl_sk - pl_phm_segment[-1], axis=1)
            b, min_b = np.argmin(d_b), np.min(d_b)
            pl_sk_segment = pl_sk[a:(b + 1)]

            # distance between phm and sk segments
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d1 = directed_hausdorff(pl_sk_segment, pl_phm_segment)[0]
                d2 = directed_hausdorff(pl_phm_segment, pl_sk_segment)[0]
                dist = max(d1, d2)

            if dist < dist_threshold and b > b_selected:
                b_selected = b
                dist_min = dist
                selected_rank = rank

        if b_selected != 0:
            l1 = polyline_length(polylines_phm[selected_rank])
            l2 = polyline_length(pl_sk[b_selected:])
            extension_factor = (l2 + l1) / l1

            res[selected_rank] = res[selected_rank] + [(round(dist_min, 4),
                                                        extension_factor,
                                                        pl_sk[b_selected:])]

    # verify that a phenomenal polyline has no more than 1 corresponding skeleton polyline.
    # And keep extension polylines in a list for optional display
    extension_polylines = []
    for k in res.keys():
        # 1 skeleton branch candidate
        if len(res[k]) == 1:
            _, extension_factor, extension_pl = res[k][0]
            res[k] = extension_factor
            extension_polylines.append(extension_pl)
        # several skeleton branch candidates
        elif len(res[k]) > 1:
            d_min = float('inf')
            selected_pl = None
            for d, extension_factor, extension_pl in res[k]:
                if d < d_min:
                    d_min = d
                    res[k] = extension_factor
                    selected_pl = extension_pl
            extension_polylines.append(selected_pl)
        # no candidate
        else:
            res[k] = None

    return res, extension_polylines


def display_leaf_extension(binary, polylines_sk, polylines_phm_2d, extension_polylines,
                           show_sk=True, show_phm=True, show_ext=True):
    image = binary[..., np.newaxis] * np.ones((binary.shape[0], binary.shape[1], 3)) * 255.
    if show_sk:
        for pl in polylines_sk:
            image = cv2.polylines(np.float32(image), [pl.astype(int).reshape((-1, 1, 2))], False, (0, 0, 255), 6)
            image = cv2.circle(image, (int(pl[-1][0]), int(pl[-1][1])), 10, (0, 0, 255), -1)
    if show_phm:
        for pl in polylines_phm_2d:
            image = cv2.polylines(np.float32(image), [pl.astype(int).reshape((-1, 1, 2))], False, (255, 0, 0), 6)
            image = cv2.circle(image, (int(pl[-1][0]), int(pl[-1][1])), 10, (255, 0, 0), -1)
    if show_ext:
        for pl in extension_polylines:
            image = cv2.polylines(np.float32(image), [pl.astype(int).reshape((-1, 1, 2))], False, (0, 255, 0), 2)
    plt.figure()
    plt.imshow(image / 255.)

def leaf_extension(vmsi, binaries, shooting_frame, display_parameters = (None, False, False, False)):
    """

    Args:
        vmsi:
        binaries: {side angle : binary image}. each image pixel equals 0 or 255.
        shooting_frame:

    Returns: vmsi object with a new 'pm_length_extended' key in the .info attribute of each leaf.

    """

    display_angle, show_sk, show_phm, show_ext = display_parameters

    # ============================================================================================================

    # compute extension for each phenomenal leaf and each camera angle. Regroup results in a dictionary.

    polylines_phm = [vmsi.get_leaf_order(k).real_longest_polyline() for k in range(1, 1 + vmsi.get_number_of_leaf())]
    angles = binaries.keys()

    binaries2 = binaries.copy()
    for angle in angles:
        binaries2[angle] = binaries2[angle] / 255.

    res = dict()
    for angle in angles:

        # 2D skeleton polylines
        polylines_sk = skeleton_branches(binaries2[angle])

        # phenomenal polylines projected in 2D
        polylines_phm_2d = [phm3d_to_px2d(pl, shooting_frame, angle) for pl in polylines_phm]

        # compute leaf extension factor for each phenomenal leaf (if a result is found)
        extension_factors, extension_polylines = compute_extension(polylines_phm_2d, polylines_sk)

        res[angle] = extension_factors

        if angle == display_angle:
            #display_leaf_extension(binaries[angle], polylines_sk, polylines_phm_2d, extension_polylines)
            full_polylines_phm = [vmsi.get_leaf_order(k).get_highest_polyline().polyline
                                  for k in range(1, 1 + vmsi.get_number_of_leaf())]
            full_polylines_phm_2d = [phm3d_to_px2d(pl, shooting_frame, angle) for pl in full_polylines_phm]
            display_leaf_extension(binaries2[angle], polylines_sk, full_polylines_phm_2d, extension_polylines,
                                   show_sk=show_sk, show_phm=show_phm, show_ext=show_ext)

    # ============================================================================================================

    # merge results to have a single extension factor (median value) for each phenomenal leaf.
    # (if no skeleton segment was found for a given phenomenal leaf, extension factor have a default value of 1.)

    for k in range(1, 1 + vmsi.get_number_of_leaf()):

        leaf_ext = [res[a][k - 1] for a in angles if res[a][k - 1] is not None]

        if vmsi.get_leaf_order(k).info['pm_label'] == 'growing_leaf':
            leaf_length = vmsi.get_leaf_order(k).info['pm_length_with_speudo_stem']
        else:
            leaf_length = vmsi.get_leaf_order(k).info['pm_length']

        if not leaf_ext:
            extension_factor = 1.
        else:
            extension_factor = np.median(leaf_ext)

        vmsi.get_leaf_order(k).info['pm_length_extended'] = leaf_length * extension_factor

    return vmsi


