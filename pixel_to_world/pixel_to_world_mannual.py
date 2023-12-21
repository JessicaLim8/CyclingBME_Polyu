import numpy as np
import csv

class C2PM:
    
    window1_lower_x_pixel_coordinate = [[562, 374], [510, 357], [460, 338], [414, 324], [375, 310], [327, 295], [286, 281],
                                        [247, 267], [209, 255], [172, 243], [139, 229], [106, 219], [72, 205], [40, 196]]
    window1_right_y_pixel_coordinate = [[563, 375], [567, 340], [571, 311], [575, 284], [579, 259], [582, 237], [585, 219],
                                        [587, 201], [590, 186], [592, 171], [594, 156], [595, 141], [597, 130], [599, 119]]
    window2_lower_x_pixel_coordinate = [[1236, 196], [1196, 212], [1156, 227], [1115, 241], [1073, 258], [1031, 274], [988, 289],
                                        [944, 306], [899, 322], [853, 339], [805, 355], [756, 374], [705, 393]]
    window2_left_y_pixel_coordinate = [[706, 392], [705, 358], [705, 329], [705, 299], [705, 273], [705, 252], [705, 230],
                                       [705, 210], [706, 193], [705, 179], [706, 164], [706, 151], [706, 139], [704, 126]]
                       
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
        for i in range(len(C2PM.window1_lower_x_pixel_coordinate)):
            x1.append(C2PM.window1_lower_x_pixel_coordinate[i][0])
            y1.append((i + 10) * 100)
        p1 = np.poly1d(np.polyfit(x1, y1, 9))
        output = p1(input)
        return output

    def window1_right_y(input):
        x2 = []
        y2 = []
        for i in range(len(C2PM.window1_right_y_pixel_coordinate)):
            x2.append(C2PM.window1_right_y_pixel_coordinate[i][1])
            y2.append(i * 100)
        p2 = np.poly1d(np.polyfit(x2, y2, 9))
        output = p2(input)
        return output

    def window2_lower_x(input):
        x3 = []
        y3 = []
        for i in range(len(C2PM.window2_lower_x_pixel_coordinate)):
            x3.append(C2PM.window2_lower_x_pixel_coordinate[i][0])
            y3.append(i * 100)
        p3 = np.poly1d(np.polyfit(x3, y3, 9))
        output = p3(input)
        return output

    def window2_left_y(input):
        x4 = []
        y4 = []
        for i in range(len(C2PM.window2_left_y_pixel_coordinate)):
            x4.append(C2PM.window2_left_y_pixel_coordinate[i][1])
            y4.append(i * 100)
        p4 = np.poly1d(np.polyfit(x4, y4, 9))
        output = p4(input)
        return output

    def window1_pixel2actual(x, y):
        # 获取底线参数 
        a_w1_lower_boundary, b_w1_lower_boundary = C2PM.find_line_equation(C2PM.window1_lower_x_pixel_coordinate[0][0], C2PM.window1_lower_x_pixel_coordinate[0][1],
                                                                          C2PM.window1_lower_x_pixel_coordinate[13][0], C2PM.window1_lower_x_pixel_coordinate[13][1])
        a_w1_right_boundary, b_w1_right_boundary = C2PM.find_line_equation(C2PM.window1_right_y_pixel_coordinate[0][0], C2PM.window1_right_y_pixel_coordinate[0][1],
                                                                          C2PM.window1_right_y_pixel_coordinate[13][0], C2PM.window1_right_y_pixel_coordinate[13][1])

        # ------------x 值： 根据扇形分区，每一条从原点出发的线都具有相同的x值
        # 两条边线
        a1_x, b1_x = C2PM.find_line_equation(C2PM.window1_lower_x_pixel_coordinate[0][0], C2PM.window1_lower_x_pixel_coordinate[0][1],
                                            C2PM.window1_right_y_pixel_coordinate[13][0], C2PM.window1_right_y_pixel_coordinate[13][1])
        a2_x, b2_x = C2PM.find_line_equation(C2PM.window1_lower_x_pixel_coordinate[13][0], C2PM.window1_lower_x_pixel_coordinate[13][1],
                                            268, 47)
        # 两条边线交叉点为原点
        origin_x = C2PM.find_cross_point(a1_x, b1_x, a2_x, b2_x)
        # 边线与目标点相连的线段
        a_x, b_x = C2PM.find_line_equation(origin_x[0], origin_x[1], x, y)
        # 上述线段与window 1 lower boundary相交点
        cross_point_x = C2PM.find_cross_point(a_x, b_x, a_w1_lower_boundary, b_w1_lower_boundary)
        actual_x = C2PM.window1_lower_x(cross_point_x[0])

        # ------------y 值： 相同原理
        # 两条边线
        a1_y, b1_y = C2PM.find_line_equation(C2PM.window1_lower_x_pixel_coordinate[0][0], C2PM.window1_lower_x_pixel_coordinate[0][1],
                                            C2PM.window1_lower_x_pixel_coordinate[13][0], C2PM.window1_lower_x_pixel_coordinate[13][1])
        a2_y, b2_y = C2PM.find_line_equation(C2PM.window1_right_y_pixel_coordinate[13][0], C2PM.window1_right_y_pixel_coordinate[13][1],
                                            268, 47)
        # 两条边线交叉点为原点
        origin_y = C2PM.find_cross_point(a1_y, b1_y, a2_y, b2_y)
        # 边线与目标点相连的线段
        a_y, b_y = C2PM.find_line_equation(origin_y[0], origin_y[1], x, y)
        # 上述线段与window 1 right boundary相交点
        cross_point_y = C2PM.find_cross_point(a_y, b_y, a_w1_right_boundary, b_w1_right_boundary)
        actual_y = C2PM.window1_right_y(cross_point_y[1])

        return actual_x, actual_y

    def window2_pixel2actual(x, y):
        # 获取底线参数
        a_w2_lower_boundary, b_w2_lower_boundary = C2PM.find_line_equation(C2PM.window2_lower_x_pixel_coordinate[0][0], C2PM.window2_lower_x_pixel_coordinate[0][1],
                                                                          C2PM.window2_lower_x_pixel_coordinate[12][0], C2PM.window2_lower_x_pixel_coordinate[12][1],)
        a_w2_left_boundary, b_w2_left_boundary = C2PM.find_line_equation(C2PM.window2_left_y_pixel_coordinate[0][0], C2PM.window2_left_y_pixel_coordinate[0][1],
                                                                          C2PM.window2_left_y_pixel_coordinate[13][0], C2PM.window2_left_y_pixel_coordinate[13][1])
        # ------------x 值： 根据扇形分区，每一条从原点出发的线都具有相同的x值
        # 两条边线
        a11_x, b11_x = C2PM.find_line_equation(C2PM.window2_lower_x_pixel_coordinate[0][0], C2PM.window2_lower_x_pixel_coordinate[0][1],
                                              1012, 30)
        a21_x, b21_x = C2PM.find_line_equation(C2PM.window2_lower_x_pixel_coordinate[12][0], C2PM.window2_lower_x_pixel_coordinate[12][1],
                                              C2PM.window2_left_y_pixel_coordinate[13][0], C2PM.window2_left_y_pixel_coordinate[13][1])
        # 两条边线交叉点为原点
        origin_x = C2PM.find_cross_point(a11_x, b11_x, a21_x, b21_x)
        # 边线与目标点相连的线段
        a_x1, b_x1 = C2PM.find_line_equation(origin_x[0], origin_x[1], x, y)
        # 上述线段与window 2 lower boundary相交点
        cross_point_x1 = C2PM.find_cross_point(a_x1, b_x1, a_w2_lower_boundary, b_w2_lower_boundary)
        actual_x = C2PM.window2_lower_x(cross_point_x1[0])

        # ------------y 值： 相同原理
        # 两条边线
        a12_y, b12_y = C2PM.find_line_equation(C2PM.window2_lower_x_pixel_coordinate[0][0], C2PM.window2_lower_x_pixel_coordinate[0][1],
                                              C2PM.window2_lower_x_pixel_coordinate[12][0], C2PM.window2_lower_x_pixel_coordinate[12][1])
        a22_y, b22_y = C2PM.find_line_equation(1012, 30,
                                              C2PM.window2_left_y_pixel_coordinate[13][0], C2PM.window2_left_y_pixel_coordinate[13][1])
        # 两条边线交叉点为原点
        origin_y = C2PM.find_cross_point(a12_y, b12_y, a22_y, b22_y)
        # 边线与目标点相连的线段
        a_y2, b_y2 = C2PM.find_line_equation(origin_y[0], origin_y[1], x, y)
        # 上述线段与window 2 lower boundary相交点
        cross_point_y2 = C2PM.find_cross_point(a_y2, b_y2, a_w2_left_boundary, b_w2_left_boundary)
        actual_y = C2PM.window2_left_y(cross_point_y2[1])

        # return cross_point_y
        return actual_x, actual_y
