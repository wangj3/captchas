import sys
sys.path.insert(0, "src")

import tensorflow as tf
import argparse
import os
import tensorflow.keras as keras
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--out_dir", default="out", type=str, help="Out dir")
    # parser.add_argument("--pretrained_model", type=str, required=True)
    pretrained_model= "test.h5.h5"
    args = parser.parse_args()

    json_file = open('test.h5.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights('test.h5.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                  metrics=['accuracy'])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(os.path.join(".", 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
