# COMP 9517 project
# python 3.7
# cv2 3.4
import cv2
import copy

CELL_COUNT_LOC = (20, 20)
CELL_MITOSIS_LOC = (20,40)
CELL_DETAILS_LOC_1 = (20, 20)
CELL_DETAILS_LOC_2 = (20, 40)
CELL_DETAILS_LOC_3 = (20, 60)

# Drawer.load integrates features from Preprocessor and Matcher
# and draw contours, trajectory, path and mark mitosis
# Drawer.sever will show desired image directly

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
            print('image number:',len(self.cells_history),'cell number:',len(cells))
            # Apply contours and ids on to image
        
        for i in range(len(self.images_history)):
            current_image = self.images_history[i] 
            current_cells = self.cells_history[i] 
            mitosis_parent_count = 0

            for c in current_cells.values():
                ##### draw with different color if split

                if (c.split_p): # If a cell is a parent in a mitosis
                    mitosis_parent_count +=1

                if_split = c.if_split()
                if if_split:
                    current_image = cv2.drawContours(current_image, c.get_contour(), -1, (255, 0, 0), 2)
                else:
                    current_image = cv2.drawContours(current_image, c.get_contour(), -1, (0, 255, 0), 2)
                current_image = cv2.putText(current_image, str(c.get_id()), c.get_centroid(), 1, 1, (0, 255, 0), 1)
                # Draw previous trajectory
                previous_positions = c.get_prev_positions()
                ##### draw with different color
                for k in range(len(previous_positions) - 1):
                    current_image = cv2.line(current_image, previous_positions[k], previous_positions[k + 1], (0, 0, 255), 1, 4)
            
            # Add a black padding on top for putting text 
            current_image = cv2.copyMakeBorder(current_image,60,0,0,0,cv2.BORDER_CONSTANT,0)
            # Show count of cells
            current_image = cv2.putText(current_image, f'Cell count: {len(current_cells)}', CELL_COUNT_LOC, 1, 1, (0, 255, 0), 1)
            current_image = cv2.putText(current_image, f'Dividing cell count: {mitosis_parent_count}', CELL_MITOSIS_LOC, 1, 1, (0, 255, 0), 1)
            # Save generated image
            self.gen_history.append(current_image)

     # Serve currently loaded image
    # id parameter allows user to input a cell ID in the terminal
    # and the Drawer will recompute the image with additional information for that cell
    def serve(self, frame, id=None):
        frame = frame -1
        image = copy.deepcopy(self.gen_history[frame])
        if id:
            cell = self.cells_history[frame][id]
            speed = cell.get_speed()
            total_dist = cell.get_total_dist()
            net_dist = cell.get_net_dist()
            confinement_ratio = cell.get_confinement_ratio()
            print(f'Speed = {speed}\nTotal distance = {total_dist}\nNet distance = {net_dist}\nConfinement ratio = {confinement_ratio}\n')
            # Add a black padding on top for putting text 
            image = cv2.copyMakeBorder(image,70,0,0,0,cv2.BORDER_CONSTANT,0)

            image = cv2.putText(image,
                                f'Cell ID {id}: Speed = {speed:.2f}',
                                CELL_DETAILS_LOC_1, 1, 1, (255, 255, 255), 1)
            image = cv2.putText(image,
                                f'Total distance = {total_dist:.2f}, Net distance = {net_dist:.2f}',
                                CELL_DETAILS_LOC_2, 1, 1, (255, 255, 255), 1)
            image = cv2.putText(image,
                                f'Confinement ratio = {confinement_ratio:.2f}',
                                CELL_DETAILS_LOC_3, 1, 1, (255, 255, 255), 1)
        return image
    
    # Returns all generated images
    def get_gen_images(self):
        return self.gen_history
    
    # Returns cell history    
    def get_cell_history(self):
        return self.cells_history
