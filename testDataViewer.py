from tkinter import *
from testDataOpener import training_data
from neuralModel import NeuralNetwork
import threading




CANVAS_HEIGHT = 420
CANVAS_WIDTH = 420

WINDOW_WIDTH = 2000
WINDOW_HEIGHT = 1000

COUNT = 0

class DrawnNumberVis:
    def __init__(self):
        self.model = NeuralNetwork()
        self.model.load_parameters()
        self.root = Tk()
        self.root.configure(bg="black")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.top_row = Frame(self.root, bg = "black")
        self.top_row.pack(pady = 30)

        self.bottom_row = Frame(self.root, bg = "black")
        self.bottom_row.pack(pady = 4)

        # top row
        self.canvas = Canvas(self.top_row, height = CANVAS_HEIGHT, width = CANVAS_WIDTH)
        self.canvas.pack(side = LEFT)

        self.number_text = Label(self.top_row, text = str(training_data[1][COUNT]), fg = "white", bg = "black", font=("Arial", 100))
        self.number_text.pack(side = RIGHT, padx = 100)

        # bottom row
        self.button = Button(self.bottom_row, text = "Next", padx = 30, pady = 10, command = self.displayNextImg, bg = "black", fg = "white", font=("Arial", 20))
        self.button.pack(side = LEFT)

        self.model_guess_text = Label(self.bottom_row, text = str(self.model_guess(training_data[0][COUNT])), fg = "white", bg = "black", font=("Arial", 80))
        self.model_guess_text.pack(side = RIGHT, padx = 100)

        self.training_data = training_data   

    def setBackround(self, colour):
        for item in [self.top_row, self.bottom_row, self.root, self.number_text, self.model_guess_text]:
            item.config(bg = colour)

    def drawNumber(self, pixelValues):
        global img
        img = PhotoImage(width=28, height=28)

        count = 0
        for y in range(28):
            for x in range(28):
                img.put(self.value_to_hex(pixelValues[count]), (x, y))
                count += 1

        # scale image to 840x840 (30x scaling)
        img_large = img.zoom(30, 30)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=img_large, anchor="nw")

        # keep reference so it doesn't get garbage collected
        self.canvas.image = img_large

    def displayNextImg(self):
        global COUNT
        self.drawNumber(training_data[0][COUNT])
        self.number_text.config(text = str(training_data[1][COUNT]))
        self.model_guess_text.config(text = str(self.model_guess(training_data[0][COUNT])))
        if str(training_data[1][COUNT]) == str(self.model_guess(training_data[0][COUNT])):
            self.setBackround("green")
        else:
            self.setBackround("red")
        COUNT += 1

    def model_guess(self, data):
        #self.model = NeuralNetwork()

        self.model.input(data)
        self.model.forward_propogation()
        return self.model.output()




    def value_to_hex(self, x):    
        x = max(0, min(1, x))  # clamp to [0,1]
        intensity = int((x) * 255)
        return f"#{intensity:02x}{intensity:02x}{intensity:02x}"
    
    def run(self):
        self.displayNextImg()
        self.root.mainloop()

if __name__ == "__main__":
    main = DrawnNumberVis()
    main.run()
