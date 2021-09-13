import cv2
import numpy as np

from openalea.maizetrack.utils import rgb_and_polylines


class Interface:
    def __init__(self):
        self.size = 1000
        self.button_size = int(self.size / 10)
        self.buttons = ['<-', '->', '-1', '+1', 'OK', 'cam', 'r=-1', 'r=10']

        # (x, y) pos for the left center of each button
        self.buttons_positions = [(0, 150 + ib * self.button_size) for ib in range(len(self.buttons))]

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
                if y_bt < y < y_bt + s and x_bt < x < x_bt + s:
                    self.current_button = button_name


    def reset_click(self):

        self.click = False
        self.current_button = ''

    def display(self, img):

        # img = (2048, 2448) rgb image of plant + polylines

        # resize for display
        img_rs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rs = cv2.resize(img_rs, (self.size, self.size))

        # display buttons
        s = self.button_size
        for (x_bt, y_bt), button_name in zip(self.buttons_positions, self.buttons):
            img_rs = cv2.rectangle(img_rs, (x_bt, y_bt), (x_bt + s, y_bt + s), (255, 255, 255), -1)
            img_rs = cv2.rectangle(img_rs, (x_bt, y_bt), (x_bt + s, y_bt + s), (0, 0, 0), 5)
            img_rs = cv2.putText(img_rs, button_name, (x_bt + int(s / 3), y_bt + int(s / 2)),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(self.window_name, img_rs.astype(np.uint8))



def annotate(plant, init=True):

    interface = Interface()

    angles = [60, 150]

    if not all([a in plant.snapshots[0].image.keys() for a in angles]):
        print('loading images..')
        for angle in angles:
            print(angle)
            plant.load_images(angle)
        print('images loaded')

    # start annotation from alignment result
    if init:
        for snapshot in plant.snapshots:
            snapshot.rank_annotation = snapshot.get_ranks()

    i_img = 0
    angle = angles[0]

    img, polylines = rgb_and_polylines(plant.snapshots[i_img], angle=angle)
    interface.display(img)

    while True:

        if interface.click:

            if interface.current_button == '->' and i_img < len(plant.snapshots) - 1:
                i_img += 1
                img, polylines = rgb_and_polylines(plant.snapshots[i_img], angle=angle)
                interface.display(img)

            elif interface.current_button == '<-' and i_img > 0:
                i_img -= 1
                img, polylines = rgb_and_polylines(plant.snapshots[i_img], angle=angle)
                interface.display(img)

            elif interface.current_button == 'cam':

                i_angle = angles.index(angle)
                if i_angle == len(angles) - 1:
                    angle = angles[0]
                else:
                    angle = angles[i_angle + 1]

                img, polylines = rgb_and_polylines(plant.snapshots[i_img], angle=angle)
                interface.display(img)

            elif interface.current_button == '':

                # conversion to (2448, 2048) scale
                y, x = interface.mouse_pos
                y, x = (np.array([y, x]) / interface.size * np.array(img.shape[:2])).astype(int)

                dists = []
                for pl in polylines:
                    d = min([np.linalg.norm(np.array([x, y]) - xy) for xy in pl])
                    dists.append(d)
                i_selected = np.argmin(dists)

                print('t')
                img, polylines = rgb_and_polylines(plant.snapshots[i_img], angle=angle, selected=i_selected)
                interface.display(img)
                interface.reset_click()
                selection = True

                while selection:

                    if interface.click:

                        if interface.current_button in ['+1', '-1', 'r=-1', 'r=10']:

                            if interface.current_button == '+1':
                                plant.snapshots[i_img].rank_annotation[i_selected] += 1
                            elif interface.current_button == '-1':
                                plant.snapshots[i_img].rank_annotation[i_selected] -= 1
                            elif interface.current_button == 'r=-1':
                                plant.snapshots[i_img].rank_annotation[i_selected] = -1
                            elif interface.current_button == 'r=10':
                                plant.snapshots[i_img].rank_annotation[i_selected] = 10

                            img, polylines = rgb_and_polylines(plant.snapshots[i_img], angle=angle, selected=i_selected)
                            interface.display(img)

                        elif interface.current_button == 'OK':

                            img, polylines = rgb_and_polylines(plant.snapshots[i_img], angle=angle)
                            interface.display(img)
                            selection = False

                        interface.reset_click()

                    k = cv2.waitKey(20) & 0xFF
                    if k == 27:
                        break

            interface.reset_click()

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    return plant


