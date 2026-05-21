import tkinter as tk
from gui import FaceRecognitionGUI

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()