import cv2

CELL_COUNT_LOC = (20, 20)
CELL_DETAILS_LOC = (20, 50)


class Drawer:
    def __init__(self, matcher):
        self.matcher = matcher
        self.image = None
        self.cells = None

    # Get next frame's image and cells array from Matcher
    def next(self):
        image, cells = self.matcher.next()
        # Apply contours and ids on to image
        ### one channel to three
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for c in cells.values():
            ##### draw with different color if split
            if_split = c.if_split()
            if if_split:
                image = cv2.drawContours(image, c.get_contour(), -1, (255, 0, 0), 1)
            else:
                image = cv2.drawContours(image, c.get_contour(), -1, (0, 255, 0), 1)
            image = cv2.putText(image, str(c.get_id()), c.get_centroid(), 1, 1, (0, 255, 0), 1)
            # Draw previous trajectory
            previous_positions = c.get_prev_positions()
            ##### draw with different color
            # if len(previous_positions) > 1:
            #     # print(previous_positions)
            #     image = cv2.drawContours(image, [previous_positions], -1, (0, 255, 0), 1)
            for i in range(len(previous_positions) - 1):
                image = cv2.line(image, previous_positions[i], previous_positions[i + 1], (0, 255, 0), 1, 4)

        # Show count of cells
        image = cv2.putText(image, f'Cell count: {len(cells)}', CELL_COUNT_LOC, 1, 1, (0, 255, 0), 1)

        self.image = image
        self.cells = cells
        ####  cells in imgaes = []

    # Serve currently loaded image
    # id parameter allows user to input a cell ID in the terminal
    # and the Drawer will recompute the image with additional information for that cell
    def serve(self, id=None):
        image = self.image
        if id:
            cell = self.cells[id]
            speed = cell.get_speed()
            total_dist = cell.get_total_dist()
            net_dist = cell.get_net_dist()
            confinement_ratio = cell.get_confinement_ratio()
            print(f'{speed}, {total_dist}, {net_dist}, {confinement_ratio}')
            image = cv2.putText(image,
                                f'Cell ID {id}: Speed = {speed:.2f}, Total distance = {total_dist:.2f}, Net distance = {net_dist:.2f}, Confinement ratio = {confinement_ratio:.2f}',
                                CELL_DETAILS_LOC, 1, 1, (0, 255, 0), 1)
        return image
