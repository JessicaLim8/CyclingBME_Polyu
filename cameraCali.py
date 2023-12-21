import math
import cannyegde
main_points = cannyedge.getEdges

def calculate_angle(current_x, current_y, desired_x, desired_y):
    # Calculate the differences in X and Y coordinates
    delta_x = desired_x - current_x
    delta_y = desired_y - current_y
    
    # Calculate the angle in radians using arctan2
    angle_rad = math.atan2(delta_y, delta_x)
    
    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


angle = calculate_angle(current_x, current_y, desired_x, desired_y)
print(f"Angle to adjust: {angle} degrees")