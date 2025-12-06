import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalAveragePooling1D, Reshape, MultiHeadAttention, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=2):
    image_input = Input(shape=input_shape, name="image_input")
    biomarker_input = Input(shape=(1,), name="biomarker_input")

    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    
    x = Reshape((1, 256))(x)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = GlobalAveragePooling1D()(attention_output)

    concat = Concatenate()([x, biomarker_input])
    
    outputs = Dense(num_classes, activation='softmax')(concat)
    
    model = Model(inputs=[image_input, biomarker_input], outputs=outputs)
    return model

def export_to_tflite(model, output_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
