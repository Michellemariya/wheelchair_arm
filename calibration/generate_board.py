# calibration/generate_board.py
import cv2
import numpy as np

def generate_charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # 9 columns, 6 rows, 25mm squares, 12.5mm markers
    board = cv2.aruco.CharucoBoard((9, 6), 0.025, 0.0125, dictionary)
    
    # Generate image at print resolution (300 DPI for A4 = ~3508x2480)
    img = board.generateImage((3508, 2480), marginSize=50)
    cv2.imwrite('calibration/charuco_board_A4.png', img)
    
    print("Board saved. Print at exact A4 size.")
    print("After printing, measure one square with a ruler.")
    print("It must be exactly 25mm. If not, adjust square_length parameter.")

if __name__ == '__main__':
    generate_charuco_board()