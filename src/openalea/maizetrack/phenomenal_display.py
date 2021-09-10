# %gui qt
import matplotlib.pyplot as plt #necessaire pour que plantGL fonctionne ??

import numpy as np
from openalea.plantgl import all as pgl


# TODO : use same PALETTE in all functions

PALETTE = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [204, 121, 167], [0, 158, 115],
     [0, 114, 178], [230, 159, 0], [140, 86, 75], [0, 255, 255], [255, 0, 100], [0, 77, 0], [100, 0, 255],
     [100, 0, 0], [0, 0, 100], [100,100, 0], [0, 100,100], [100, 0, 100], [0, 0, 0], [255, 100, 100]])

PALETTE = 3 * list(PALETTE) + [[255, 255, 255]]



def generic_plot(objects, col=False):

    palette = np.array(
        [[255,255,255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
         [0, 100, 255], [0, 255, 100], [100, 0, 255], [100, 255, 0], [255, 0, 100], [255, 100, 0]])

    shapes = []

    for i, obj in enumerate(objects):

        size = 4
        if col:
            r, g, b = palette[i]
        else:
            r, g, b = 255, 255, 255
        m = pgl.Material(pgl.Color3(int(r), int(g), int(b)))
        for x, y, z in obj:
            b = pgl.Box(size, size, size)
            b = pgl.Translated(x + i, y + i, z + i, b)
            b = pgl.Shape(b, m)
            shapes.append(b)

    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)





def plot2(vg):
    shapes = voxelgrid_to_pgl(vg)
    print(len(shapes), ' shapes to plot')
    shapes += image_to_pgl('ZM4363/bin/side/90.png', (2048,2448))
    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)


def image_to_pgl(path, points):
    indices = [(0, 1, 2, 3)]
    background = pgl.QuadSet(points, indices)
    texture = pgl.ImageTexture(path)
    background.texCoordList = [(0, 0), (1, 0), (1, 1), (0, 1)]
    background.texCoordIndexList = [(0, 1, 2, 3)]
    plant2d = pgl.Shape(background, texture)
    shapes = [plant2d]
    #scene = pgl.Scene(shapes)
    return shapes

def voxelgrid_to_pgl(vg,recul=0.0):
    dezoom = 1
    remonte = 0
    size = vg.voxels_size/dezoom
    shapes = []
    m = pgl.Material(pgl.Color3(60, 60, 60))
    #m = pgl.Material(pgl.Color3(30, 170, 0))
    for x, y, z in vg.voxels_position:
        b = pgl.Box(size, size, size)
        vx = pgl.Translated(x/dezoom, y/dezoom, (z + remonte)/dezoom, b)
        vx = pgl.Translated(0, recul, 0, vx)
        vx = pgl.Shape(vx, m)
        shapes.append(vx)
    return shapes


def skeleton_to_pgl(skeleton,r_cylindre=3):
    shapes = []
    col1 = pgl.Material(pgl.Color3(255, 150, 0))
    col2 = pgl.Material(pgl.Color3(255, 0, 0))
    for organ in skeleton.segments:
        segment = organ.polyline

        for k in range(len(segment) - 1):
            # arguments cylindre : centre1, centre2, rayon, nbre segments d'un cercle.
            cyl = pgl.Extrusion(pgl.Polyline([segment[k], segment[k + 1]]), pgl.Polyline2D.Circle(r_cylindre, 8))
            cyl.solid = True  # rajoute 2 cercles aux extremites du cylindre
            cyl = pgl.Shape(cyl, col1)
            shapes.append(cyl)

        x, y, z = segment[-1]
        tip = pgl.Sphere(30)
        tip = pgl.Translated(x, y, z, tip)
        tip = pgl.Shape(tip, col2)
        shapes.append(tip)
    return shapes


def mesh_to_pgl(vertices,faces):
    h = 700 # hauteur entre le sol et la camera (mm)
    shapes = []
    for i1,i2,i3 in faces:
        points = np.array([vertices[i1], vertices[i2], vertices[i3]])
        points += np.array([0,0,h])
        triangle = pgl.TriangleSet(points, [(0, 1, 2)])
        shapes.append(pgl.Shape(triangle))
    return shapes



def vmsi_polylines_to_pgl(vmsi, li='all', coli='same', only_mature=False):
    shapes = []
    h = 700 # - vmsi.get_stem().info['pm_z_base']
    col2 = pgl.Material(pgl.Color3(0, 0, 0))
    col3 = pgl.Material(pgl.Color3(255, 0, 0))

    if li=='all':
        leaves = vmsi.get_leafs()
    else:
        leaves = [vmsi.get_leaf_order(k) for k in li]

    if only_mature:
        leaves = [leaf for leaf in leaves if leaf.info['pm_label'][0]=='m']

    for l in range(len(leaves)):
        leaf = leaves[l]

        #segment = leaf.get_highest_polyline().polyline
        segment = leaf.real_longest_polyline()
        #col1 = pgl.Material(pgl.Color3(np.random.randint(255), np.random.randint(255), np.random.randint(255)))


        if leaf.info['pm_label'][0]=='m':
            col1 = pgl.Material(pgl.Color3(0, 128, 255))
        elif leaf.info['pm_label'][0]=='g':
            #col1 = pgl.Material(pgl.Color3(100, 100, 255))
            col1 = pgl.Material(pgl.Color3(255, 140, 0))
        else:
            col1 = pgl.Material(pgl.Color3(255, 255, 255))


        palette = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,255,255]])
        palette = list(palette) + list((palette*0.66).astype(int)) + list((palette*0.33).astype(int))

        if coli != 'same':
            r, g, b = palette[coli[l]]
            col1 = pgl.Material(pgl.Color3(int(r), int(g), int(b)))


        for k in range(len(segment) - 1):
            # arguments cylindre : centre1, centre2, rayon, nbre segments d'un cercle.
            pos1 = np.array(segment[k]) + np.array([0, 0, h])
            pos2 = np.array(segment[k+1]) + np.array([0, 0, h])
            cyl = pgl.Extrusion(pgl.Polyline([pos1, pos2]), pgl.Polyline2D.Circle(8, 8))
            cyl.solid = True  # rajoute 2 cercles aux extremites du cylindre
            cyl = pgl.Shape(cyl, col1)
            shapes.append(cyl)

        # leaf tip
        #x, y, z = segment[-1]
        #tip = pgl.Sphere(10)
        #tip = pgl.Translated(x, y, z+h, tip)
        #tip = pgl.Shape(tip, col2)
        #shapes.append(tip)

        # leaf base
        #x, y, z = leaf.info['pm_position_base']
        #tip2 = pgl.Sphere(10)
        #tip2 = pgl.Translated(x, y, z+h, tip2)
        #tip2 = pgl.Shape(tip2, col3)
        #shapes.append(tip2)

    # stem
    segment = vmsi.get_stem().get_highest_polyline().polyline
    for k in range(len(segment) - 1):
        # arguments cylindre : centre1, centre2, rayon, nbre segments d'un cercle.
        pos1 = np.array(segment[k]) + np.array([0, 0, h])
        pos2 = np.array(segment[k + 1]) + np.array([0, 0, h])
        cyl = pgl.Extrusion(pgl.Polyline([pos1, pos2]), pgl.Polyline2D.Circle(8, 8))
        cyl.solid = True  # rajoute 2 cercles aux extremites du cylindre
        cyl = pgl.Shape(cyl, col2)
        shapes.append(cyl)

    return shapes



def plot_sk(skeleton):
    shapes = skeleton_to_pgl(skeleton)
    print(len(shapes), ' shapes to plot')
    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)


def plot_vg_sk(vg,sk,img=False):

    # skeleton + voxelgrid
    shapes_vg = voxelgrid_to_pgl(vg,recul=-100)
    shapes_sk = skeleton_to_pgl(sk, r_cylindre=10)
    shapes = shapes_vg + shapes_sk
    if img:
        # image 90deg
        x, z = (2048, 2448)
        recul = 1000
        points = [(-x / 2, -recul, -z / 2), (x / 2, -recul, -z / 2), (x / 2, -recul, z / 2),
                  (-x / 2, -recul, z / 2)]  # image dans le plan x/z, centrée au point 0,0,0
        shapes += image_to_pgl('ZM4363/bin/side/90.png', points)

        # image 0deg
        points = [(-x / 2, -recul+x, -z / 2), (-x / 2, -recul, -z / 2), (-x / 2, -recul, z / 2),
                  (-x / 2, -recul+x, z / 2)]  # image dans le plan x/z, centrée au point 0,0,0
        shapes += image_to_pgl('ZM4363/bin/side/0.png', points)

    print(len(shapes), ' shapes to plot')
    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)

#plot_sq()


# TODO : remove l, col
def plot_vmsi(vmsi_list, l=[], col=[], only_mature=False):
    shapes = []
    for i in range(len(vmsi_list)):
        vmsi = vmsi_list[i]

        if l==[]:
            li = 'all'
        else:
            li = l[i]

        if col==[]:
            coli = 'same'
        else:
            coli = col[i]

        shapes += vmsi_polylines_to_pgl(vmsi, li, coli, only_mature)
    print(len(shapes), ' shapes to plot')
    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)






def plot_vmsi_voxel(vmsi, ranks=None):

    size = vmsi.voxels_size
    shapes = []

    organs = [vmsi.get_stem()] + \
             [vmsi.get_unknown()] + \
             [vmsi.get_leaf_order(k + 1) for k in range(len(vmsi.get_leafs()))]

    #bricolage
    stem_pl = np.array(list(vmsi.get_stem().get_highest_polyline().polyline))
    insertions = [voxel_insertion(leaf, stem_pl) for leaf in vmsi.get_mature_leafs()]
    h_vx = max([h[2] for h in insertions if h is not None])

    for k, organ in enumerate(organs):

        if k != -100: # no stem

            if ranks is None:

                # unified color
                # c1, c2, c3 = 20, 90, 10

                # color depending on organ type
                if organ.info['pm_label'] == 'stem':
                    c1, c2, c3 = 0, 0, 0
                elif organ.info['pm_label'] == 'growing_leaf':
                    # c1, c2, c3 = 230, 94, 0
                    c1, c2, c3 = 255, 140, 0

                elif organ.info['pm_label'] == 'mature_leaf':
                    # c1, c2, c3 = 0, 0, 255
                    # c1, c2, c3 = 34, 113, 178
                    c1, c2, c3 = 0, 128, 255
                else:
                    c1, c2, c3 = 0, 0, 0

            else:

                if organ.info['pm_label'] == 'stem':
                    c1, c2, c3 = 0, 0, 0
                elif organ.info['pm_label'] in ['growing_leaf', 'mature_leaf']:
                    i_leaf = ranks[organ.info['pm_leaf_number'] - 1]
                    c1, c2, c3 = PALETTE[i_leaf]
                    if i_leaf == -1:
                        c1, c2, c3 = 100, 100, 100
                else:
                    c1, c2, c3 = 0, 0, 0

            m = pgl.Material(pgl.Color3(int(c1), int(c2), int(c3)))


            for x, y, z in organ.voxels_position():

                #bricolage, pour faire partir les growing au niveau de la derniere mature
                if organ.info['pm_label'] in ['stem', 'unknown']:
                    if z > h_vx:
                        m = pgl.Material(pgl.Color3(int(255), int(140), int(0)))
                        #m = pgl.Material(pgl.Color3(int(100), int(100), int(100)))
                        #m = pgl.Material(pgl.Color3(int(0), int(255), int(0)))
                    else:
                        m = pgl.Material(pgl.Color3(int(c1), int(c2), int(c3)))

                b = pgl.Box(size, size, size)
                vx = pgl.Translated(x, y, z, b)
                #vx = pgl.Translated(0, recul, 0, vx)
                vx = pgl.Shape(vx, m)
                shapes.append(vx)

    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)











def plot_vg(vg):
    shapes = voxelgrid_to_pgl(vg)
    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)


def plot_comparaison(vmsi, model_scene, model_json):
    #shapes = mesh_to_pgl(vertices, faces)
    shapes = vmsi_polylines_to_pgl(vmsi)

    # cereals leaf tips
    for polyline in model_json['leaf_polylines']:
        col2 = pgl.Material(pgl.Color3(255, 0, 0))
        x,y,z,_ = polyline[-1]
        tip = pgl.Sphere(3)
        tip = pgl.Translated(x, y, z, tip)
        tip = pgl.Shape(tip, col2)
        shapes.append(tip)

    print(len(shapes), 'shapes to plot')
    shapes = pgl.Scene(shapes)
    scene = pgl.Scene([shapes,model_scene])
    pgl.Viewer.display(scene)


def plot_simplemaize(model_scene, model_json):
    shapes = []

    for polyline in model_json['leaf_polylines']:
        col2 = pgl.Material(pgl.Color3(255, 0, 0))
        x,y,z,_ = polyline[-1]
        tip = pgl.Sphere(3)
        tip = pgl.Translated(x, y, z, tip)
        tip = pgl.Shape(tip, col2)
        shapes.append(tip)

    print(len(shapes), 'shapes to plot')
    shapes = pgl.Scene(shapes)
    scene = pgl.Scene([shapes,model_scene])
    pgl.Viewer.display(scene)


def plot_leaves(leaves,cluster,stem=False):

    #palette = np.array(
    #    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
    #     [0,100,255], [0,255,100], [100,0,255], [100,255,0], [255,0,100], [255,100,0]])

    #palette = list(palette) + list((palette * 0.3).astype(int)) + \
    #          list((palette * 0.15).astype(int)) + [[255, 255, 255]]


    #palette = np.array([[0, 0, 0], [230, 159, 0], [86, 180, 233], [0, 158, 115],
    #                    [240, 228, 66], [0, 114, 178], [213, 94, 0], [204, 121, 167]])
    #palette = 3 * list(palette) + [[255, 255, 255]]

    shapes = []
    h = 700  # - vmsi.get_stem().info['pm_z_base']

    for i in range(len(leaves)):

        if stem == False:
            segment = leaves[i].real_longest_polyline() # for leaf
        else:
            segment = leaves[i].get_highest_polyline().polyline # for stem
        r, g, b = PALETTE[cluster[i]]
        col = pgl.Material(pgl.Color3(int(r), int(g), int(b)))

        r = 10 * np.random.random()
        for k in range(len(segment) - 1):
            # arguments cylindre : centre1, centre2, rayon, nbre segments d'un cercle.
            pos1 = np.array(segment[k]) + np.array([0, 0, h + r])
            pos2 = np.array(segment[k + 1]) + np.array([0, 0, h + r])
            cyl = pgl.Extrusion(pgl.Polyline([pos1, pos2]), pgl.Polyline2D.Circle(5, 8))
            cyl.solid = True  # rajoute 2 cercles aux extremites du cylindre
            cyl = pgl.Shape(cyl, col)
            shapes.append(cyl)

    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)



def plot_snapshot(snapshot, colored=True, ranks=None):

    size = 15

    leaves = snapshot.leaves
    if colored and ranks is None:
        ranks = snapshot.get_ranks()
    print(ranks)

    z_stem = max([l.info['pm_z_base'] for l in snapshot.get_mature_leafs()])

    shapes = []
    h = 700  # - vmsi.get_stem().info['pm_z_base']
    for i in range(len(leaves)):

        leaf = leaves[i]
        if leaf.info['pm_label'] == 'growing_leaf':
            segment = leaves[i].get_highest_polyline().polyline
            segment = [x for x in segment if x[2] > z_stem]
        else:
            segment = leaves[i].real_longest_polyline() # for leaf

        if colored:
            if ranks[i] == -1:
                r, g, b = 80, 80, 80
            else:
                r, g, b = PALETTE[ranks[i]]
            col = pgl.Material(pgl.Color3(int(r), int(g), int(b)))
        else:
            if leaf.info['pm_label'][0] == 'm':
                col = pgl.Material(pgl.Color3(0, 128, 255))
            elif leaf.info['pm_label'][0] == 'g':
                col = pgl.Material(pgl.Color3(255, 140, 0))

        for k in range(len(segment) - 1):
            # arguments cylindre : centre1, centre2, rayon, nbre segments d'un cercle.
            pos1 = np.array(segment[k]) + np.array([0, 0, h])
            pos2 = np.array(segment[k + 1]) + np.array([0, 0, h])
            cyl = pgl.Extrusion(pgl.Polyline([pos1, pos2]), pgl.Polyline2D.Circle(size, 8))
            cyl.solid = True  # rajoute 2 cercles aux extremites du cylindre
            cyl = pgl.Shape(cyl, col)
            shapes.append(cyl)

    # stem
    col2 = pgl.Material(pgl.Color3(0, 0, 0))
    segment = snapshot.get_stem().get_highest_polyline().polyline
    segment = [x for x in segment if x[2] <= z_stem]
    for k in range(len(segment) - 1):
        # arguments cylindre : centre1, centre2, rayon, nbre segments d'un cercle.
        pos1 = np.array(segment[k]) + np.array([0, 0, h])
        pos2 = np.array(segment[k + 1]) + np.array([0, 0, h])
        cyl = pgl.Extrusion(pgl.Polyline([pos1, pos2]), pgl.Polyline2D.Circle(int(size * 1.3), 8))
        cyl.solid = True  # rajoute 2 cercles aux extremites du cylindre
        cyl = pgl.Shape(cyl, col2)
        shapes.append(cyl)

    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)