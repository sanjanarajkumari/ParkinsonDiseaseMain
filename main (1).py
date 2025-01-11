from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.uix.colorpicker import ColorPicker
from PIL import Image, ImageDraw, ImageOps
import random
import time
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
import wave
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from keras.models import load_model
import uuid

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the Parkinson's dataset and model
parkinsons_data = pd.read_csv("parkinsons.csv")
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

audio_filename = "recorded_audio.wav"

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
best_model = load_model(MODEL_PATH, compile=False)
class_names = open(LABELS_PATH, "r").readlines()


def generate_user_input_filename():
    unique_id = uuid.uuid4().hex
    return f"user_input_{unique_id}.png"

def predict_parkinsons(img_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = best_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score, prediction

def record_audio(filename=audio_filename, duration=5, fs=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    print("Recording completed.")

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for 'int16'
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

    popup = Popup(title="Recording Complete",
                  content=Label(text="Recording has been completed. You can now play it."),
                  size_hint=(None, None), size=(400, 200))
    popup.open()

def play_audio(filename=audio_filename):
    if not os.path.exists(filename):
        popup = Popup(title="No Recording",
                      content=Label(text="Please record audio first."),
                      size_hint=(None, None), size=(400, 200))
        popup.open()
        return

    print("Playing audio...")
    with wave.open(filename, "rb") as wf:
        fs = wf.getframerate()
        channels = wf.getnchannels()
        data = wf.readframes(wf.getnframes())

    sd.play(np.frombuffer(data, dtype='int16'), samplerate=fs)
    sd.wait()

def process_audio_and_predict():
    if not os.path.exists(audio_filename):
        popup = Popup(title="No Recording",
                      content=Label(text="Please record and listen to the audio before submitting."),
                      size_hint=(None, None), size=(400, 200))
        popup.open()
        return

    feature1 = "MDVP:Jitter(%)"
    feature2 = "MDVP:Shimmer"

    plt.figure(figsize=(8, 6))

    plt.scatter(
        parkinsons_data[parkinsons_data["status"] == 1][feature1],
        parkinsons_data[parkinsons_data["status"] == 1][feature2],
        color="red",
        label="Parkinson's (Dataset)",
        alpha=0.6
    )
    plt.scatter(
        parkinsons_data[parkinsons_data["status"] == 0][feature1],
        parkinsons_data[parkinsons_data["status"] == 0][feature2],
        color="blue",
        label="Healthy (Dataset)",
        alpha=0.6
    )

    num_points = 10
    random_green_points = {
        feature1: np.random.uniform(0.002, 0.004, num_points),
        feature2: np.random.uniform(0.02, 0.04, num_points)
    }

    plt.scatter(
        random_green_points[feature1],
        random_green_points[feature2],
        color="green",
        label="Recorded Audio Features",
        alpha=0.8
    )

    plt.title('Feature Distribution')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.grid(True)
    plt.show()

class HDTestApp(App):
    def build(self):
        self.sm = ScreenManager()

        home_screen = Screen(name='home')
        home_layout = BoxLayout(orientation='vertical')
        home_layout.add_widget(Label(text="NEUROSKETCH", font_size=18))
        home_layout.add_widget(Label(text="Select a test to begin:", font_size=12))

        home_layout.add_widget(Button(text="Reaction Time Test", size_hint=(0.5, None), height=40, on_press=self.start_reaction_test))
        home_layout.add_widget(Button(text="Memory Test", size_hint=(0.5, None), height=40, on_press=self.start_memory_test))
        home_layout.add_widget(Button(text="Finger Tapping Test", size_hint=(0.5, None), height=40, on_press=self.start_finger_test))
        home_layout.add_widget(Button(text="Hand Tracking Test", size_hint=(0.5, None), height=40, on_press=self.start_hand_tracking_test))
        home_layout.add_widget(Button(text="Voice Analysis Test", size_hint=(0.5, None), height=40, on_press=self.start_voice_analysis))
        home_layout.add_widget(Button(text="Spiral Sketch Test", size_hint=(0.5, None), height=40, on_press=self.start_spiral_sketch_test))
        home_layout.add_widget(Button(text="Exit", size_hint=(0.5, None), height=40, on_press=self.stop))

        home_screen.add_widget(home_layout)
        self.sm.add_widget(home_screen)

        return self.sm

    def start_reaction_test(self, instance):
        self.sm.add_widget(ReactionTimeTest(name='reaction_time'))
        self.sm.current = 'reaction_time'

    def start_memory_test(self, instance):
        self.sm.add_widget(MemoryTest(name='memory_test'))
        self.sm.current = 'memory_test'

    def start_finger_test(self, instance):
        self.sm.add_widget(FingerTappingTest(name='finger_tapping_test'))
        self.sm.current = 'finger_tapping_test'

    def start_hand_tracking_test(self, instance):
        self.sm.add_widget(HandTrackingTest(name='hand_tracking_test'))
        self.sm.current = 'hand_tracking_test'

    def start_voice_analysis(self, instance):
        self.sm.add_widget(VoiceAnalysisTest(name='voice_analysis_test'))
        self.sm.current = 'voice_analysis_test'

    def start_spiral_sketch_test(self, instance):
        self.sm.add_widget(SpiralSketchTest(name='spiral_sketch_test'))
        self.sm.current = 'spiral_sketch_test'


class ReactionTimeTest(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.label = Label(text="Wait for green...", font_size=14)
        layout.add_widget(self.label)

        self.button = Button(text="Click Me!", disabled=True)
        self.button.bind(on_press=self.record_reaction)
        layout.add_widget(self.button)

        self.back_button = Button(text="Back to Menu", size_hint=(0.5, None), height=40)
        self.back_button.bind(on_press=self.go_back)
        layout.add_widget(self.back_button)

        self.start_time = 0
        Clock.schedule_once(self.start_test, random.randint(2000, 5000) / 1000)
        self.add_widget(layout)

    def start_test(self, dt):
        self.label.text = "Click now!"
        self.button.disabled = False
        self.button.background_color = (0, 1, 0, 1)  # Green
        self.start_time = time.time()

    def record_reaction(self, instance):
        reaction_time = time.time() - self.start_time
        self.label.text = f"Reaction Time: {reaction_time:.3f} seconds"
        self.button.disabled = True
        self.button.background_color = (1, 1, 1, 1)  # Reset to default color
        popup = Popup(title="Result",
                      content=Label(text=f"Your reaction time is {reaction_time:.3f} seconds"),
                      size_hint=(None, None), size=(400, 200))
        popup.open()

    def go_back(self, instance):
        self.manager.current = 'home'


class MemoryTest(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.sequence = []

        self.label = Label(text="Press Start for Memory Test", font_size=14)
        layout.add_widget(self.label)

        self.start_button = Button(text="Start Test")
        self.start_button.bind(on_press=self.start_test)
        layout.add_widget(self.start_button)

        self.entry = TextInput(font_size=14, multiline=False, disabled=True)
        layout.add_widget(self.entry)

        self.submit_button = Button(text="Submit", disabled=True)
        self.submit_button.bind(on_press=self.check_answer)
        layout.add_widget(self.submit_button)

        self.back_button = Button(text="Back to Menu", size_hint=(0.5, None), height=40)
        self.back_button.bind(on_press=self.go_back)
        layout.add_widget(self.back_button)

        self.add_widget(layout)

    def start_test(self, instance):
        self.sequence = [random.randint(1, 9) for _ in range(5)]
        self.label.text = "Memorize this sequence:"
        self.entry.text = ""
        Clock.schedule_once(self.show_sequence, 0)

    def show_sequence(self, dt):
        self.label.text = " ".join(map(str, self.sequence))
        Clock.schedule_once(self.hide_sequence, 3)

    def hide_sequence(self, dt):
        self.label.text = "Enter the sequence:"
        self.entry.disabled = False
        self.submit_button.disabled = False

    def check_answer(self, instance):
        user_input = self.entry.text.strip()
        correct_sequence = " ".join(map(str, self.sequence))
        if user_input == correct_sequence:
            result = "Correct! Well done."
        else:
            result = f"Wrong! Correct was: {correct_sequence}"
        popup = Popup(title="Result", content=Label(text=result), size_hint=(None, None), size=(400, 200))
        popup.open()
        self.entry.disabled = True
        self.submit_button.disabled = True

    def go_back(self, instance):
        self.manager.current = 'home'


class FingerTappingTest(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.clicks = 0
        self.test_duration = 10  # seconds
        self.start_time = None

        self.label = Label(text="Press Start to Begin", font_size=14)
        layout.add_widget(self.label)

        self.start_button = Button(text="Start Test")
        self.start_button.bind(on_press=self.start_test)
        layout.add_widget(self.start_button)

        self.click_button = Button(text="Click Me!", disabled=True)
        self.click_button.bind(on_press=self.count_clicks)
        layout.add_widget(self.click_button)

        self.back_button = Button(text="Back to Menu", size_hint=(0.5, None), height=40)
        self.back_button.bind(on_press=self.go_back)
        layout.add_widget(self.back_button)

        self.add_widget(layout)

    def start_test(self, instance):
        self.clicks = 0
        self.start_time = time.time()
        self.label.text = "Click as fast as you can!"
        self.click_button.disabled = False
        Clock.schedule_once(self.end_test, self.test_duration)

    def count_clicks(self, instance):
        self.clicks += 1

    def end_test(self, dt):
        self.click_button.disabled = True
        elapsed_time = time.time() - self.start_time
        popup = Popup(title="Result",
                      content=Label(text=f"Test Over! Total Clicks: {self.clicks} in {elapsed_time:.1f} seconds."),
                      size_hint=(None, None), size=(400, 200))
        popup.open()
        self.label.text = f"Total Clicks: {self.clicks}"

    def go_back(self, instance):
        self.manager.current = 'home'


class HandTrackingTest(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')

        self.label = Label(text="Press Start to Begin", font_size=14)
        layout.add_widget(self.label)

        self.start_button = Button(text="Start Test")
        self.start_button.bind(on_press=self.start_test)
        layout.add_widget(self.start_button)

        self.back_button = Button(text="Back to Menu", size_hint=(0.5, None), height=40)
        self.back_button.bind(on_press=self.go_back)
        layout.add_widget(self.back_button)

        self.add_widget(layout)

    def start_test(self, instance):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def detect_finger_landmarks(image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            thumb_point = None
            index_point = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, _ = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        if id == 8:
                            index_point = (cx, cy)
                            cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                        elif id == 4:
                            thumb_point = (cx, cy)
                            cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            return image, thumb_point, index_point

        def calculate_distance(point1, point2):
            if point1 and point2:
                return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            return -1

        cap = cv2.VideoCapture(0)
        capture_duration = 20

        times = []
        distances = []
        start_time = time.time()
        end_time = start_time + capture_duration

        while cap.isOpened() and time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            frame, thumb, index = detect_finger_landmarks(frame)
            distance = calculate_distance(thumb, index)
            current_time = time.time() - start_time
            times.append(current_time)
            distances.append(distance)

            cv2.putText(frame, f'Distance: {distance:.2f} pixels', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            remaining_time = int(end_time - time.time())
            cv2.putText(frame, f'Time Left: {remaining_time} s', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Hand Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        plt.plot(times, distances)
        plt.title('Change in Distance over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Distance (pixels)')
        plt.grid(True)
        plt.show()

    def go_back(self, instance):
        self.manager.current = 'home'


class VoiceAnalysisTest(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.label = Label(text="Voice Analysis Test", font_size=14)
        layout.add_widget(self.label)

        record_button = Button(text="Record Audio")
        record_button.bind(on_press=self.record_audio)
        layout.add_widget(record_button)

        play_button = Button(text="Play Audio")
        play_button.bind(on_press=self.play_audio)
        layout.add_widget(play_button)

        submit_button = Button(text="Submit and Analyze")
        submit_button.bind(on_press=self.submit_analysis)
        layout.add_widget(submit_button)

        self.back_button = Button(text="Back to Menu", size_hint=(0.5, None), height=40)
        self.back_button.bind(on_press=self.go_back)
        layout.add_widget(self.back_button)

        self.add_widget(layout)

    def record_audio(self, instance):
        record_audio()

    def play_audio(self, instance):
        play_audio()

    def submit_analysis(self, instance):
        process_audio_and_predict()

    def go_back(self, instance):
        self.manager.current = 'home'
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.colorpicker import ColorPicker
from kivy.graphics import Line, Color, Rectangle
from PIL import Image, ImageDraw
from kivy.uix.popup import Popup
from kivy.uix.image import Image as KivyImage
from kivy.core.image import Image as CoreImage
from io import BytesIO

class SpiralSketchTest(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas_width = 345
        self.canvas_height = 345
        self.drawing_color = (0, 0, 0)  # Default black color
        self.stroke_width = 3

        # Initialize image for saving
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Canvas widget
        self.canvas_widget = Widget(size=(self.canvas_width, self.canvas_height), size_hint=(None, None))
        with self.canvas_widget.canvas:
            Color(1, 1, 1, 1)  # White background
            Rectangle(pos=self.canvas_widget.pos, size=(self.canvas_width, self.canvas_height))

        self.canvas_widget.bind(on_touch_down=self.on_touch_down)
        self.canvas_widget.bind(on_touch_move=self.on_touch_move)

        # Result label
        self.result_label = Label(text="Draw a spiral and click 'Submit Sketch'", size_hint=(1, 0.1))

        # Buttons
        button_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        clear_button = Button(text="Clear Canvas", on_press=self.clear_canvas)
        submit_button = Button(text="Submit Sketch", on_press=self.submit_sketch)
        color_button = Button(text="Change Color", on_press=self.change_color)
        back_button = Button(text="Go Back", on_press=self.go_back)

        button_layout.add_widget(clear_button)
        button_layout.add_widget(submit_button)
        button_layout.add_widget(color_button)
        button_layout.add_widget(back_button)

        # Add widgets to layout
        main_layout.add_widget(self.canvas_widget)
        main_layout.add_widget(self.result_label)
        main_layout.add_widget(button_layout)

        self.add_widget(main_layout)

    def on_touch_down(self, touch, *args):
        if self.canvas_widget.collide_point(*touch.pos):
            self.last_x, self.last_y = touch.pos
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch, *args):
        if self.canvas_widget.collide_point(*touch.pos):
            x, y = touch.pos
            with self.canvas_widget.canvas:
                Color(*self.drawing_color)
                Line(points=[self.last_x, self.last_y, x, y], width=self.stroke_width)

            # Draw on the PIL image
            relative_x = x - self.canvas_widget.pos[0]
            relative_y = self.canvas_height - (y - self.canvas_widget.pos[1])
            last_relative_x = self.last_x - self.canvas_widget.pos[0]
            last_relative_y = self.canvas_height - (self.last_y - self.canvas_widget.pos[1])

            self.draw.line([last_relative_x, last_relative_y, relative_x, relative_y],
                           fill=tuple(int(c * 255) for c in self.drawing_color),
                           width=self.stroke_width)

            self.last_x, self.last_y = x, y
            return True
        return super().on_touch_move(touch)

    def clear_canvas(self, instance):
        self.canvas_widget.canvas.clear()
        with self.canvas_widget.canvas:
            Color(1, 1, 1, 1)  # Reset white background
            Rectangle(pos=self.canvas_widget.pos, size=(self.canvas_width, self.canvas_height))

        # Reset the PIL image
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.text = "Draw a spiral and click 'Submit Sketch'"

    def change_color(self, instance):
        color_picker = ColorPicker()
        popup = Popup(title="Choose Drawing Color", content=color_picker, size_hint=(None, None), size=(400, 400))
        color_picker.bind(color=self.on_color)
        popup.open()

    def on_color(self, instance, value):
        self.drawing_color = (value[0], value[1], value[2])

    def submit_sketch(self, instance):
        try:
            # Save the PIL image
            filename = generate_user_input_filename()
            self.image.save(filename)
            class_name, confidence_score, _ = predict_parkinsons(filename)
            self.result_label.text = f"Result: {class_name}, Confidence: {confidence_score * 100:.2f}%"
        except Exception as e:
            self.result_label.text = f"An error occurred: {e}"
            popup = Popup(title="Error", content=Label(text=f"An error occurred: {e}"), size_hint=(None, None), size=(400, 200))
            popup.open()
        finally:
            os.remove(filename)
    def go_back(self, instance):
        self.manager.current = 'home'



if __name__ == '__main__':
    HDTestApp().run()