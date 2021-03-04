import cv2
from tflite_runtime.interpreter import Interpreter
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('-m',
                    '--model_file',
                    default='/deps/miniVGGNET.tflite',
                    help='.tflite model to be executed')

parser.add_argument('-i',
                    '--image_file',
                    default='/deps/model_no_cpu_temp.tflite',
                    help='.tflite model to be executed')

args = parser.parse_args()


#invoke tflite API
interpreter = Interpreter(model_path=args['model_file'])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = cv2.imread(args['image_file'])
image = np.expand_dims(cv2.resize(image, (64, 64)), axis=0)
in_tensor = np.float32(image/255.)

interpreter.set_tensor(input_details[0]['index'], in_tensor)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

with open("/deps/class_names.pickle", 'rb') as f:
    class_names = pickle.load(f)

print("The fruit is: ", class_names[np.argmax(output_data)])