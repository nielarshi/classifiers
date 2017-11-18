import numpy as np

# Input data - [x value, y value, bias term]
'''
    Point 1 = < 7, 11, Δ >
    Point 2 = < 11, 11, Δ > 
    Point 3 = < 13, 11, Δ >
    Point 4 = < 8, 10, Δ > 
    Point 5 = < 9, 9, Δ > 
    Point 6 = < 15, 9, o > 
    Point 7 = < 7, 7, Δ > 
    Point 8 = < 15, 7, o > 
    Point 9 = < 7, 5, Δ > 
    Point 10 = < 13, 5, o > 
    Point 11 = < 14, 4, o > 
    Point 12 = < 9, 3, o > 
    Point 13 = < 11, 3, o > 
    Point 14 = < 15, 3, o > 
    Point 15 = < 10, 7, Δ > 
'''
sample_data_points_with_bias = np.array([
    [7, 11, -1],
    [11, 11, -1],
    [13, 11, -1],
    [8, 10, -1],
    [9, 9, -1],
    [15, 9, -1],
    [7, 7, -1],
    [15, 7, -1],
    [7, 5, -1],
    [13, 5, -1],
    [14, 4, -1],
    [9, 3, -1],
    [11, 3, -1],
    [15, 3, -1],
    [10, 7, -1]
])

sample_data_points_wo_bias = np.array([
    [7, 11],
    [11, 11],
    [13, 11],
    [8, 10],
    [9, 9],
    [15, 9],
    [7, 7],
    [15, 7],
    [7, 5],
    [13, 5],
    [14, 4],
    [9, 3],
    [11, 3],
    [15, 3],
    [10, 7]
])

known_classifications = np.array(
    [1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1]
)