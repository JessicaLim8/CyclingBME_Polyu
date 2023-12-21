import numpy as np
import csv

class C2PA:
    file = csv.reader(open('x_w1_pixelCoordinate', 'r'))
    window1_lower_x_pixel_coordinate = []
    for i in file:
        window1_lower_x_pixel_coordinate.append(i)
    for i in range(len(window1_lower_x_pixel_coordinate)):
        for j in [0, 1]:
            window1_lower_x_pixel_coordinate[i][j] = int(window1_lower_x_pixel_coordinate[i][j])
    
    file = csv.reader(open('y_w1_pixelCoordinate', 'r'))
    window1_right_y_pixel_coordinate = []
    for i in file:
        window1_right_y_pixel_coordinate.append(i)
    for i in range(len(window1_right_y_pixel_coordinate)):
        for j in [0, 1]:
            window1_right_y_pixel_coordinate[i][j] = int(window1_right_y_pixel_coordinate[i][j])
    
    file = csv.reader(open('x_w2_pixelCoordinate', 'r'))
    window2_lower_x_pixel_coordinate = []
    for i in file:
        window2_lower_x_pixel_coordinate.append(i)
    for i in range(len(window2_lower_x_pixel_coordinate)):
        for j in [0, 1]:
            window2_lower_x_pixel_coordinate[i][j] = int(window2_lower_x_pixel_coordinate[i][j])
    
    file = csv.reader(open('y_w2_pixelCoordinate', 'r'))
    window2_left_y_pixel_coordinate = []
    for i in file:
        window2_left_y_pixel_coordinate.append(i)
    for i in range(len(window2_left_y_pixel_coordinate)):
        for j in [0, 1]:
            window2_left_y_pixel_coordinate[i][j] = int(window2_left_y_pixel_coordinate[i][j])

#     window1_lower_x_pixel_coordinate = [[514, 395], [463, 370], [414, 348], [370, 325], [326, 305], [284, 286],
#                                         [244, 267], [207, 251], [170, 232], [134, 217], [99, 199], [67, 185], [37, 169]]
#     window1_right_y_pixel_coordinate = [[514, 394], [528, 360], [539, 328], [550, 300], [559, 275], [566, 251],
#                                         [574, 232], [582, 211], [587, 195], [593, 179], [599, 165], [604, 152],
#                                         [609, 139], [613, 126]]
#     window2_lower_x_pixel_coordinate = [[1121, 200], [1092, 208], [1066, 217], [1034, 228], [996, 240], [966, 250],
#                                         [933, 259], [898, 270], [860, 282], [824, 293], [789, 303], [748, 316]]
#     window2_left_y_pixel_coordinate =[[747, 318], [747, 294], [747, 278], [746, 262], [746, 246], [746, 232], [745, 220], [745, 202]]
                       
    def find_line_equation(x1, y1, x2, y2):
        a = (y2-y1)/(x2-x1)
        b = y1-x1*a
        return a, b

    def find_cross_point(a1, b1, a2, b2):
        x = (b2-b1)/(a1-a2)
        y = a1*x+b1
        return x, y

    def window1_lower_x(input):
        x1 = []
        y1 = []
        for i in range(len(C2PA.window1_lower_x_pixel_coordinate)):
            x1.append(C2PA.window1_lower_x_pixel_coordinate[i][0])
            y1.append((i + 10) * 100)
        p1 = np.poly1d(np.polyfit(x1, y1, 9))
        output = p1(input)
        return output

    def window1_right_y(input):
        x2 = []
        y2 = []
        for i in range(len(C2PA.window1_right_y_pixel_coordinate)):
            x2.append(C2PA.window1_right_y_pixel_coordinate[i][1])
            y2.append(i * 100)
        p2 = np.poly1d(np.polyfit(x2, y2, 9))
        output = p2(input)
        return output

    def window2_lower_x(input):
        x3 = []
        y3 = []
        for i in range(len(C2PA.window2_lower_x_pixel_coordinate)):
            x3.append(C2PA.window2_lower_x_pixel_coordinate[i][0])
            y3.append(i * 100)
        p3 = np.poly1d(np.polyfit(x3, y3, 9))
        output = p3(input)
        return output

    def window2_left_y(input):
        x4 = []
        y4 = []
        for i in range(len(C2PA.window2_left_y_pixel_coordinate)):
            x4.append(C2PA.window2_left_y_pixel_coordinate[i][1])
            y4.append(i * 100)
        p4 = np.poly1d(np.polyfit(x4, y4, 9))
        output = p4(input)
        return output

    def window1_pixel2actual(x, y):
        # 获取底线参数 
        a_w1_lower_boundary, b_w1_lower_boundary = C2PA.find_line_equation(C2PA.window1_lower_x_pixel_coordinate[0][0], C2PA.window1_lower_x_pixel_coordinate[0][1],
                                                                          C2PA.window1_lower_x_pixel_coordinate[13][0], C2PA.window1_lower_x_pixel_coordinate[13][1])
        a_w1_right_boundary, b_w1_right_boundary = C2PA.find_line_equation(C2PA.window1_right_y_pixel_coordinate[0][0], C2PA.window1_right_y_pixel_coordinate[0][1],
                                                                          C2PA.window1_right_y_pixel_coordinate[13][0], C2PA.window1_right_y_pixel_coordinate[13][1])

        # ------------x 值： 根据扇形分区，每一条从原点出发的线都具有相同的x值
        # 两条边线
        a1_x, b1_x = C2PA.find_line_equation(C2PA.window1_lower_x_pixel_coordinate[0][0], C2PA.window1_lower_x_pixel_coordinate[0][1],
                                            C2PA.window1_right_y_pixel_coordinate[13][0], C2PA.window1_right_y_pixel_coordinate[13][1])
        a2_x, b2_x = C2PA.find_line_equation(C2PA.window1_lower_x_pixel_coordinate[13][0], C2PA.window1_lower_x_pixel_coordinate[13][1],
                                            270, 46)
        # 两条边线交叉点为原点
        origin_x = C2PA.find_cross_point(a1_x, b1_x, a2_x, b2_x)
        # 边线与目标点相连的线段
        a_x, b_x = C2PA.find_line_equation(origin_x[0], origin_x[1], x, y)
        # 上述线段与window 1 lower boundary相交点
        cross_point_x = C2PA.find_cross_point(a_x, b_x, a_w1_lower_boundary, b_w1_lower_boundary)
        actual_x = C2PA.window1_lower_x(cross_point_x[0])

        # ------------y 值： 相同原理
        # 两条边线
        a1_y, b1_y = C2PA.find_line_equation(C2PA.window1_lower_x_pixel_coordinate[0][0], C2PA.window1_lower_x_pixel_coordinate[0][1],
                                            C2PA.window1_lower_x_pixel_coordinate[13][0], C2PA.window1_lower_x_pixel_coordinate[13][1])
        a2_y, b2_y = C2PA.find_line_equation(C2PA.window1_right_y_pixel_coordinate[13][0], C2PA.window1_right_y_pixel_coordinate[13][1],
                                            270, 46)
        # 两条边线交叉点为原点
        origin_y = C2PA.find_cross_point(a1_y, b1_y, a2_y, b2_y)
        # 边线与目标点相连的线段
        a_y, b_y = C2PA.find_line_equation(origin_y[0], origin_y[1], x, y)
        # 上述线段与window 1 right boundary相交点
        cross_point_y = C2PA.find_cross_point(a_y, b_y, a_w1_right_boundary, b_w1_right_boundary)
        actual_y = C2PA.window1_right_y(cross_point_y[1])

        return actual_x, actual_y

    def window2_pixel2actual(x, y):
        # 获取底线参数
        a_w2_lower_boundary, b_w2_lower_boundary = C2PA.find_line_equation(C2PA.window2_lower_x_pixel_coordinate[0][0], C2PA.window2_lower_x_pixel_coordinate[0][1],
                                                                          C2PA.window2_lower_x_pixel_coordinate[12][0], C2PA.window2_lower_x_pixel_coordinate[12][1],)
        a_w2_left_boundary, b_w2_left_boundary = C2PA.find_line_equation(C2PA.window2_left_y_pixel_coordinate[0][0], C2PA.window2_left_y_pixel_coordinate[0][1],
                                                                          C2PA.window2_left_y_pixel_coordinate[13][0], C2PA.window2_left_y_pixel_coordinate[13][1])
        # ------------x 值： 根据扇形分区，每一条从原点出发的线都具有相同的x值
        # 两条边线
        a11_x, b11_x = C2PA.find_line_equation(C2PA.window2_lower_x_pixel_coordinate[0][0], C2PA.window2_lower_x_pixel_coordinate[0][1],
                                              1007, 30)
        a21_x, b21_x = C2PA.find_line_equation(C2PA.window2_lower_x_pixel_coordinate[12][0], C2PA.window2_lower_x_pixel_coordinate[12][1],
                                              C2PA.window2_left_y_pixel_coordinate[13][0], C2PA.window2_left_y_pixel_coordinate[13][1])
        # 两条边线交叉点为原点
        origin_x = C2PA.find_cross_point(a11_x, b11_x, a21_x, b21_x)
        # 边线与目标点相连的线段
        a_x1, b_x1 = C2PA.find_line_equation(origin_x[0], origin_x[1], x, y)
        # 上述线段与window 2 lower boundary相交点
        cross_point_x1 = C2PA.find_cross_point(a_x1, b_x1, a_w2_lower_boundary, b_w2_lower_boundary)
        actual_x = C2PA.window2_lower_x(cross_point_x1[0])

        # ------------y 值： 相同原理
        # 两条边线
        a12_y, b12_y = C2PA.find_line_equation(C2PA.window2_lower_x_pixel_coordinate[0][0], C2PA.window2_lower_x_pixel_coordinate[0][1],
                                              C2PA.window2_lower_x_pixel_coordinate[12][0], C2PA.window2_lower_x_pixel_coordinate[12][1])
        a22_y, b22_y = C2PA.find_line_equation(1007, 30,
                                              C2PA.window2_left_y_pixel_coordinate[13][0], C2PA.window2_left_y_pixel_coordinate[13][1])
        # 两条边线交叉点为原点
        origin_y = C2PA.find_cross_point(a12_y, b12_y, a22_y, b22_y)
        # 边线与目标点相连的线段
        a_y2, b_y2 = C2PA.find_line_equation(origin_y[0], origin_y[1], x, y)
        # 上述线段与window 2 lower boundary相交点
        cross_point_y2 = C2PA.find_cross_point(a_y2, b_y2, a_w2_left_boundary, b_w2_left_boundary)
        actual_y = C2PA.window2_left_y(cross_point_y2[1])

        # return cross_point_y
        return actual_x, actual_y
