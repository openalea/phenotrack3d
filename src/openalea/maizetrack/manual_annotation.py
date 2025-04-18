"""
A cv2 draft tool to annotate leaf ranks on a time-series of 3D maize plant segmentations.

It can be used to build a ground-truth dataset to evaluate leaf tracking performances, or as manual tool to easily
refine imperfect tracking outputs.

It was developed to work with the image system of the Phenoarch platform : it would probably need a few adaptations
to work with other systems.
"""

import cv2
import numpy as np
from openalea.maizetrack.display import PALETTE


class Interface:
    def __init__(self):
        self.size = 1000
        self.button_size = int(self.size / 12)
        self.buttons = ['<-', '->', '-1', '+1', 'OK', 'cam', 'r=0', 'r=10', '+1 all', '-1 all']

        # (x, y) pos for the left center of each button
        self.buttons_positions = [(0, 50 + ib * self.button_size) for ib in range(len(self.buttons))]

        self.mouse_pos = (0, 0)
        self.current_button = ''
        self.click = False

        self.window_name = 'window'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.check_button_click)

    def check_button_click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.mouse_pos = (y, x)
            self.click = True

            s = self.button_size
            for (x_bt, y_bt), button_name in zip(self.buttons_positions, self.buttons):
                if y_bt < y < y_bt + s and x_bt < x < x_bt + int(1.5 * s):
                    self.current_button = button_name

    def reset_click(self):

        self.click = False
        self.current_button = ''

    def display(self, img):

        # image = (2048, 2448) rgb image of plant + polylines

        # resize for display
        img_rs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rs = cv2.resize(img_rs, (self.size, self.size))

        # display buttons
        s = self.button_size
        for (x_bt, y_bt), button_name in zip(self.buttons_positions, self.buttons):
            img_rs = cv2.rectangle(img_rs, (x_bt, y_bt), (x_bt + int(1.5 * s), y_bt + s), (255, 255, 255), -1)
            img_rs = cv2.rectangle(img_rs, (x_bt, y_bt), (x_bt + int(1.5 * s), y_bt + s), (0, 0, 0), 5)
            img_rs = cv2.putText(img_rs, button_name, (x_bt + int(s / 6), y_bt + int(s / 2)),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(self.window_name, img_rs.astype(np.uint8))


def annotate(annot):
    """
    Run an interface to annotate leaf ranks of a TrackedPlant object.
    Press "Esc" key to exit
    """

    def _image(annot, task, angle):
        metainfo = annot[task]['metainfo']
        image = annot[task]['images'][angle]
        leaves_info = annot[task]['leaves_info']
        leaves_pl = [polylines[angle] for polylines in annot[task]['leaves_pl']]
        return rgb_and_polylines(image, leaves_pl, leaves_info, metainfo)

    tasks = list(annot.keys())
    angles = list(annot[tasks[0]]['images'].keys())
    i_task, i_angle = 0, 0

    img_dimension = annot[tasks[0]]['images'][angles[0]].shape[:2]

    interface = Interface()
    interface.display(_image(annot, tasks[i_task], angles[i_angle]))

    while True:

        if interface.click:

            # ===== action = change task ==============================================================================

            if interface.current_button == '->' and i_task < len(tasks) - 1:
                i_task += 1
                interface.display(_image(annot, tasks[i_task], angles[i_angle]))

            elif interface.current_button == '<-' and i_task > 0:
                i_task -= 1
                interface.display(_image(annot, tasks[i_task], angles[i_angle]))

            # ===== action = change view angle ========================================================================

            elif interface.current_button == 'cam':
                i_angle = i_angle + 1 if i_angle < len(angles) - 1 else 0
                interface.display(_image(annot, tasks[i_task], angles[i_angle]))

            # ==== action = change all leaf ranks =====================================================================

            elif interface.current_button == '+1 all':
                for leaf in annot[tasks[i_task]]['leaves_info']:
                    leaf['rank'] += 1
                interface.display(_image(annot, tasks[i_task], angles[i_angle]))

            elif interface.current_button == '-1 all':
                for leaf in annot[tasks[i_task]]['leaves_info']:
                    leaf['rank'] -= 1
                interface.display(_image(annot, tasks[i_task], angles[i_angle]))

            # ===== action = select a leaf ============================================================================

            elif interface.current_button == '':

                # conversion to (2448, 2048) scale
                y, x = interface.mouse_pos
                y, x = (np.array([y, x]) / interface.size * np.array(img_dimension)).astype(int)

                dists = []
                polylines = [pl[angles[i_angle]] for pl in annot[tasks[i_task]]['leaves_pl']]
                for pl in polylines:
                    d = min([np.linalg.norm(np.array([x, y]) - xy) for xy in pl])
                    dists.append(d)
                i_selected = np.argmin(dists)
                selected_leaf = annot[tasks[i_task]]['leaves_info'][i_selected]
                selected_leaf['selected'] = True

                interface.display(_image(annot, tasks[i_task], angles[i_angle]))
                interface.reset_click()

                while selected_leaf['selected']:

                    if interface.click:

                        # ===== action = modify leaf rank =============================================================

                        if interface.current_button in ['+1', '-1', 'r=0', 'r=10']:
                            if interface.current_button == '+1':
                                selected_leaf['rank'] += 1
                            elif interface.current_button == '-1':
                                selected_leaf['rank'] -= 1
                            elif interface.current_button == 'r=0':
                                selected_leaf['rank'] = 0
                            elif interface.current_button == 'r=10':
                                selected_leaf['rank'] = 10
                            interface.display(_image(annot, tasks[i_task], angles[i_angle]))

                        # ===== action = validate the modification  ===================================================

                        elif interface.current_button == 'OK':
                            selected_leaf['selected'] = False
                            interface.display(_image(annot, tasks[i_task], angles[i_angle]))

                        interface.reset_click()

                    k = cv2.waitKey(20) & 0xFF
                    if k == 27:
                        break

            interface.reset_click()

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


def rgb_and_polylines(image, leaves_pl, leaves_info, metainfo):

    image_pl = image.copy()

    for pl, leaf in zip(leaves_pl, leaves_info):

        # col = [0, 0, 0] if c == -2 else [int(x) for x in PALETTE[c]]
        col = [int(x) for x in PALETTE[leaf['rank'] - 1]]
        border_col = (0, 0, 0) if leaf['rank'] == 0 else (255, 255, 255)

        ds = 1 if not leaf['selected'] else 3

        image_pl = cv2.polylines(np.float32(image_pl),
                                 [pl.astype(int).reshape((-1, 1, 2))], False, border_col, 10 * ds)

        image_pl = cv2.polylines(np.float32(image_pl),
                                 [pl.astype(int).reshape((-1, 1, 2))], False, col, 7 * ds)

        # tip if mature
        if leaf['mature']:
            pos = (int(pl[-1][0]), int(pl[-1][1]))
            image_pl = cv2.circle(np.float32(image_pl), pos, 20, (0, 0, 0), -1)

        # rank number
        pos = (int(pl[-1][0]), int(pl[-1][1]))
        image_pl = cv2.putText(image_pl, str(leaf['rank']), pos, cv2.FONT_HERSHEY_SIMPLEX,
                               3, (0, 0, 0), 4, cv2.LINE_AA)

    # write date
    if metainfo is not None:
        text = 'plantid {} / task {} ({})'.format(metainfo.pot, metainfo.task, metainfo.daydate)
        cv2.putText(image_pl, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 10, cv2.LINE_AA)

    return image_pl
