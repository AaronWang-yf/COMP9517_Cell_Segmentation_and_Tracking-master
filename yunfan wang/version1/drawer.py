import cv2
import copy

CELL_COUNT_LOC = (20, 20)
CELL_DETAILS_LOC = (20, 50)

class Drawer:
    def __init__(self, matcher, preprocessor):
        self.matcher = matcher
        self.preprocessor = preprocessor
        self.images_history = []
        self.gen_history = []        
        self.cells_history = []

    # Load all images and cells from Matcher
    def load(self):
        while self.preprocessor.status:
            image, cells = self.matcher.next(self.cells_history)
            ### one channel to three
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.images_history.append(image)
            self.cells_history.append(copy.deepcopy(cells))
              
            # Apply contours and ids on to image
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
                for i in range(len(previous_positions) - 1):
                    image = cv2.line(image, previous_positions[i], previous_positions[i + 1], (0, 0, 255), 1, 4)

            # Show count of cells
            image = cv2.putText(image, f'Cell count: {len(cells)}', CELL_COUNT_LOC, 1, 1, (0, 255, 0), 1)

            # Save generated image
            self.gen_history.append(image)

    # Serve currently loaded image
    # id parameter allows user to input a cell ID in the terminal
    # and the Drawer will recompute the image with additional information for that cell
    def serve(self, frame, id=None):
        image = self.images_history[frame]
        if id:
            cell = self.cells_history[frame][id]
            speed = cell.get_speed()
            total_dist = cell.get_total_dist()
            net_dist = cell.get_net_dist()
            confinement_ratio = cell.get_confinement_ratio()
            print(f' Speed = {speed}\nTotal distance = {total_dist}\nNet distance = {net_dist}\nConfinement ratio = {confinement_ratio}\n')
            image = cv2.putText(image,
                                f'Cell ID {id}: Speed = {speed:.2f}, Total distance = {total_dist:.2f}, Net distance = {net_dist:.2f}, Confinement ratio = {confinement_ratio:.2f}',
                                CELL_DETAILS_LOC, 1, 1, (0, 255, 0), 1)
        return image
    
    # Returns all generated images
    def get_gen_images(self):
        return self.gen_history
    
    # Returns cell history    
    def get_cell_history(self):
        return self.cells_history
