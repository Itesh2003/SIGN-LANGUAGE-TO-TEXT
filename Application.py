import numpy as np
import cv2
import os
import time
import operator
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
from spellchecker import SpellChecker
from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# Application :
class Application:
    def __init__(self):
        self.spell = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        # Load models
        self.load_models()
        
        self.ct = {'blank': 0}
        self.blank_flag = 0
        
        for i in ascii_uppercase:
            self.ct[i] = 0
        
        print("Loaded model from disk")
        
        # Set up GUI
        self.setup_gui()

        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def load_models(self):
        self.loaded_models = []
        model_names = [
            "model_new",
            "model-bw_dru",
            "model-bw_tkdi",
            "model-bw_smn"
        ]

        for model_name in model_names:
            json_file = open(f"Models/{model_name}.json", "r")
            model_json = json_file.read()
            json_file.close()

            loaded_model = model_from_json(model_json)
            loaded_model.load_weights(f"Models/{model_name}.h5")
            self.loaded_models.append(loaded_model)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root)  # Word
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            # Get suggestions using pyspellchecker
            predicts = self.spell.candidates(self.word)

            # Update buttons with suggestions
            suggestions = list(predicts)
            if len(suggestions) > 0:
                self.bt1.config(text=suggestions[0], font=("Courier", 20))
            else:
                self.bt1.config(text="")

            if len(suggestions) > 1:
                self.bt2.config(text=suggestions[1], font=("Courier", 20))
            else:
                self.bt2.config(text="")

            if len(suggestions) > 2:
                self.bt3.config(text=suggestions[2], font=("Courier", 20))
            else:
                self.bt3.config(text="")

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        predictions = []

        for model in self.loaded_models:
            result = model.predict(test_image.reshape(1, 128, 128, 1))
            predictions.append(result)

        # LAYER 1
        prediction = {chr(i + 65): predictions[0][0][i + 1] for i in range(26)}
        prediction['blank'] = predictions[0][0][0]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # LAYER 2
        if self.current_symbol in ['D', 'R', 'U']:
            prediction = {}
            prediction['D'] = predictions[1][0][0]
            prediction['R'] = predictions[1][0][1]
            prediction['U'] = predictions[1][0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {}
            prediction['D'] = predictions[2][0][0]
            prediction['I'] = predictions[2][0][1]
            prediction['K'] = predictions[2][0][2]
            prediction['T'] = predictions[2][0][3]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['M', 'N', 'S']:
            prediction = {}
            prediction['M'] = predictions[3][0][0]
            prediction['N'] = predictions[3][0][1]
            prediction['S'] = predictions[3][0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            if prediction[0][0] == 'S':
                self.current_symbol = prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 0:
                    self.ct[i] = 0

            if self.blank_flag == 0:
                self.blank_flag = self.ct['blank']
            self.blank_flag -= 1
            if self.blank_flag <= 0:
                self.blank_flag = 0
                self.ct['blank'] = 0
                if self.current_symbol != 'blank':
                    self.word += self.current_symbol
                    self.str += self.current_symbol

    def action1(self):
        predicts = self.spell.candidates(self.word)
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            self.str += next(iter(predicts))  # Get first suggestion

    def action2(self):
        predicts = self.spell.candidates(self.word)
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            self.str += list(predicts)[1]  # Get second suggestion

    def action3(self):
        predicts = self.spell.candidates(self.word)
        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            self.str += list(predicts)[2]  # Get third suggestion

    def destructor(self):
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
