from datetime import datetime
import math

class DataPoint:

    def __init__(self, x = 0, y = 0, acc = 0):
        self.x = x
        self.y = y
        self.accuracy = acc
        self.time = datetime.utcnow()

    def getX(self):
        return self.x
    
    def getY(self):
        return self.y

    def getAccuracy(self):
        return self.accuracy

    def getTime(self):
        return self.time
    
    def setX(self, x):
        self.x = x
    
    def setY(self, y):
        self.y = y
    
    def setAccuracy(self, acc):
        self.accuracy = acc
    
    def setDate(self, date):
        self.date = date

    def isSamePointAcc(self, point, xps, yps):
        sec = (point.getTime() - self.time).total_seconds()
        # if abs(self.x - point.getX()) > (xps * sec):
        #     return False
        # if abs(self.y - point.getY()) > (yps * sec):
        #     return False
        distance = math.hypot(point.getX() - self.getX(), point.getY() - self.getY())
        #exp_dist = math.hypot(yps, xps) * sec

        #error = abs(distance - exp_dist) / exp_dist
        error = abs(distance)
        print(f"error: {error} points: {point.getX()}, {point.getY()} and point: {self.getX()}, {self.getY()}")
        return error
