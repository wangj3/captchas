#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy

import argparse
# import tensorflow as tf
import tflite_runtime.interpreter as tflite

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")


    with open(args.output, 'w') as output_file:
        # json_file = open(args.model_name+'.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = keras.models.model_from_json(loaded_model_json)
        # model.load_weights(args.model_name+'.h5')
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        #               metrics=['accuracy'])

        for x in os.listdir(args.captcha_dir):
            # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = (numpy.array(rgb_data) / 255.0).astype(numpy.float32)
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])
            interpreter = tflite.Interpreter(model_path="model.tflite")
            # convert to grayscale
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            input = 0.299 * r + 0.587 * g + 0.114 * b

            # input has nowof  shape (70, 175)
            # we modify dimensions to match model's input
            input = numpy.expand_dims(input, 0)
            # input = numpy.expand_dims(input, -1)
            # input is now of shape (batch_size, 70, 175, 1)
            # output will have shape (batch_size, 4, 26)

            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], input)
            interpreter.invoke()

            # predict and get the output
            output = interpreter.get_tensor(output_details[0]['index'])
            # now get labels
            labels_indices = numpy.argmax(output, axis=2)

            available_chars = "abcdefghijklmnopqrstuvwxyz"

            def decode(li):
                result = []
                for char in li:
                    result.append(available_chars[char])
                return "".join(result)

            decoded_label = [decode(x) for x in labels_indices][0]

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])

            print("For file {}, the output is {}".format(x, output_data))
            print("For file {}, the output2 is {}".format(x, interpreter.get_tensor(output_details[1]['index'])))

            output_file.write(x + ", " + output_data + "\n")
            # prediction = model.predict(image)
            # output_file.write(x + ", " + decode(captcha_symbols, prediction) + "\n")

            print('Classified ' + x)

if __name__ == '__main__':
    main()
