import os
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import tensorflow as tf


class ObjectDetectionApp(App):
    def build(self):
        self.img1 = Widget()
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        btn1 = Button(text='Start Detection')
        btn1.bind(on_press=self.start_detection)
        layout.add_widget(btn1)

  
        self.interpreter = tf.lite.Interpreter(model_path="detection_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        return layout

    def start_detection(self, instance):
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 1.0/33.0) # 초당 33프레임

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            
            input_image = cv2.resize(frame, (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
            input_image = np.expand_dims(input_image, axis=0)


            input_image = input_image.astype(np.float32)

            input_image /= 255.0

      
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            self.interpreter.invoke()

          
            detected_objects = self.interpreter.get_tensor(self.output_details[0]['index'])


            frame = cv2.flip(frame, 0)
            buf = frame.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.canvas.clear()
            with self.img1.canvas:
                self.img1.texture = image_texture

if __name__ == '__main__':
    ObjectDetectionApp().run()
