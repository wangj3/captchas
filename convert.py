# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import tflite_runtime.interpreter as tfliteInterpreter
class TestModel(tf.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
    def add(self, x):
        '''
        Simple method that accepts single input 'x' and returns 'x' + 4.
        '''
        # Name the output 'result' for convenience.
        return {'result' : x + 4}


SAVED_MODEL_PATH = '.'
TFLITE_FILE_PATH = 'test.h5.tflite'
if __name__ == '__main__':
    # Save the model
    module = TestModel()
    # You can omit the signatures argument and a default signature name will be
    # created with name 'serving_default'.
    tf.saved_model.save(
        module, SAVED_MODEL_PATH,
        signatures={'my_signature':module.add.get_concrete_function()})

    # Convert the model using TFLiteConverter

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    tflite_model = converter.convert()
    with open(TFLITE_FILE_PATH, 'wb') as f:
        f.write(tflite_model)

    # Load the TFLite model in TFLite Interpreter
    interpreter = tfliteInterpreter.Interpreter(TFLITE_FILE_PATH)
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    my_signature = interpreter.get_signature_runner()



    # my_signature is callable with input as arguments.
    output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
    # 'output' is dictionary with all outputs from the inference.
    # In this case we have single output 'result'.
    print(output['result'])

# Press the green button in the gutter to run the script.

    # model = tf.keras.models.load_model('test.h5.h5')
    # lite = tf.lite.Interpreter(model_path="test.h5.tflite")
    # lite.allocate_tensors()
    #
    # inputs = lite.get_input_details()[0]
    # outputs = lite.get_output_details()[0]
