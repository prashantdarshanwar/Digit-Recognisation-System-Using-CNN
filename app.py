import tkinter as tk
from PIL import Image, ImageDraw
import os
import numpy as np

# Create folders if not exist
for i in range(10):
    os.makedirs(f"my_digits/{i}", exist_ok=True)

class DigitCollector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Data Collector")

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.label = tk.Label(self.root, text="Enter Digit Label (0-9):")
        self.label.pack()

        self.entry = tk.Entry(self.root)
        self.entry.pack()

        self.save_btn = tk.Button(self.root, text="Save", command=self.save_digit)
        self.save_btn.pack()

        self.clear_btn = tk.Button(self.root, text="Clear", command=self.clear)
        self.clear_btn.pack()

        self.root.mainloop()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill="white")
        self.draw.ellipse([x, y, x+10, y+10], fill="white")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)

    def save_digit(self):
        digit = self.entry.get()

        if digit.isdigit() and 0 <= int(digit) <= 9:
            folder = f"my_digits/{digit}"
            count = len(os.listdir(folder))
            filename = f"{folder}/{count}.png"

            img_resized = self.image.resize((28, 28))
            img_resized.save(filename)

            print(f"Saved: {filename}")
            self.clear()
        else:
            print("Enter digit between 0-9")

DigitCollector()
