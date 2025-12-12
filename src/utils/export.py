import torch
import tensorflow as tf
import os

def export_to_tflite(keras_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model exported to {output_path}")

def export_to_onnx(torch_model, input_shape, output_path):
    dummy_input = torch.randn(*input_shape)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        torch_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")
