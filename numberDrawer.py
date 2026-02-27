from tkinter import *
from math import *

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 900


def sigmoid(z):
    return 1 / (1 + exp(-z))

class NumberDrawer:
    def __init__(self):
        self.root = Tk()
        self.root.configure(bg="white")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.canvas = Canvas(self.root, height = 280, width = 280, bg = "black")
        self.canvas.pack(pady = 100)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)


    def draw(self, event):
        cx = event.x
        cy = event.y
        radius = 10

        drawn = set()
        
        for y in range(cy - radius, cy + radius + 1):
            y_offset = y - cy
           
            try:
                x_span = sqrt(radius**2 - y_offset**2)
            except ValueError:
                continue  
            x_left = cx - x_span
            x_right = cx + x_span

            y_snap = floor(y / 10) * 10
            for x in range(floor(x_left / 10) * 10, floor(x_right / 10) * 10 + 10, 10):
                x_snap = x

                if (x_snap, y_snap) not in drawn:
                    distance = (sqrt((cx - x_snap)**2 + (cy - y_snap)**2)) 
                    p = 3
                    distance_norm = 1 - (1 - (distance/sqrt(800)))**p

                    self.canvas.create_rectangle(x_snap, y_snap, x_snap + 10, y_snap + 10, fill=self.value_to_hex(distance_norm))      
                    drawn.add((x_snap, y_snap))

    def value_to_hex(self, x):    
        x = max(0, min(1, x))  # clamp to [0,1]
        intensity = int((x) * 255)
        return f"#{intensity:02x}{intensity:02x}{intensity:02x}"



    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    numberDrawer = NumberDrawer()
    numberDrawer.run()

    