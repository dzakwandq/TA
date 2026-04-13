from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, ZeroPadding2D
from keras.layers import Add, UpSampling2D, Lambda
from keras.optimizers import Adam, SGD
from keras.layers import Activation, MaxPool2D, Concatenate
from keras import backend as K
from keras.models import load_model, model_from_json
import tensorflow as tf
import cv2
import numpy as np
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom", name="resize_bilinear_44_33")
def resize_bilinear_44_33(x):
    return tf.image.resize(x, size=(44, 33), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize_bilinear_87_66")
def resize_bilinear_87_66(x):
    return tf.image.resize(x, size=(87, 66), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize_bilinear_174_132")
def resize_bilinear_174_132(x):
    return tf.image.resize(x, size=(174, 132), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize_bilinear_682_512")
def resize_bilinear_682_512(x):
    return tf.image.resize(x, size=(682, 512), method=tf.image.ResizeMethod.BILINEAR)

@register_keras_serializable(package="Custom", name="resize_output_shape_44_33")
def resize_output_shape_44_33(input_shape):
    return (input_shape[0], 44, 33, input_shape[3])

@register_keras_serializable(package="Custom", name="resize_output_shape_87_66")
def resize_output_shape_87_66(input_shape):
    return (input_shape[0], 87, 66, input_shape[3])

@register_keras_serializable(package="Custom", name="resize_output_shape_174_132")
def resize_output_shape_174_132(input_shape):
    return (input_shape[0], 174, 132, input_shape[3])

@register_keras_serializable(package="Custom", name="resize_output_shape_682_512")
def resize_output_shape_682_512(input_shape):
    return (input_shape[0], 682, 512, input_shape[3])


# Defining the function for each layer

# ENCODER
def lm1_block(input, num_filters):
  x = BatchNormalization()(input)
  # x = ZeroPadding2D(padding=1)(x)
  x = Conv2D(num_filters, 3, (2,2), padding='same')(x) # filters, kernel, strides
  x = BatchNormalization()(x)
  u = Activation('relu')(x)
  # x = ZeroPadding2D(padding=1)(u)
  x = MaxPool2D((2, 2))(u)

  return x, u

def lm2a_block(input, num_filters):
  x = BatchNormalization()(input)
  u = Activation('relu')(x)
  x = ZeroPadding2D(padding=1)(u)
  x = Conv2D(num_filters, 3, strides=(1,1), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = ZeroPadding2D(padding=1)(x)
  x = Conv2D(num_filters, 3, strides=(1,1), padding='same')(x)

  u = ZeroPadding2D(padding=1)(u)
  u = Conv2D(num_filters, 3, strides=(1,1), padding='same')(u)
  u = ZeroPadding2D(padding=1)(u)
  a = Add()([x, u])

  return a

def lm2b_block(input, num_filters):
  x = BatchNormalization()(input)
  u = Activation('relu')(x)
  # x = ZeroPadding2D(padding=1)(u)
  x = Conv2D(num_filters, 3, strides=(1,1), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # x = ZeroPadding2D(padding=1)(x)
  x = Conv2D(num_filters, 3, strides=(2,2), padding='same')(x)

  # u = ZeroPadding2D(padding=2)(u)
  u = Conv2D(num_filters, 3, strides=(2,2), padding='same')(u)
  a = Add()([x, u])

  return a

def lm3_block(input, num_filters):
  x = BatchNormalization()(input)
  x = Activation('relu')(x)
  # x = ZeroPadding2D(padding=1)(x)
  x = Conv2D(num_filters, 3, strides=(1,1), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # x = ZeroPadding2D(padding=1)(x)
  x = Conv2D(num_filters, 3, strides=(1,1), padding='same')(x)

  a = Add()([input, x])

  return a

# DECODER
def lm4_block(input, skip_features, num_filters):
  x = Concatenate()([input, skip_features])
  x = UpSampling2D((2, 2))(x)

  return x

def lm5_block(input, num_filters):
  x = Conv2D(num_filters, 3, strides=(1,1), padding='same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  return x

def lm6_block(input, num_filters):
  x = UpSampling2D(size=(2,2))(input)
  x = Conv2D(num_filters, 3, strides=(1,1), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(num_filters, 3, strides=(1,1), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  return x

input_shape = (682, 512, 3)
print(input_shape)

def build_unbcsm(input_shape):
  inputs = Input(input_shape)

  # ENCODER - LAYER 1
  l1a, ru = lm1_block(inputs, 64)
  l1b = lm2a_block(l1a, 64)
  l1c = lm3_block(l1b, 64)
  l1d = lm3_block(l1c, 64)

  # ENCODER - LAYER 2
  l2a = lm2b_block(l1d, 128)
  l2b = lm3_block(l2a, 128)
  l2c = lm3_block(l2b, 128)
  l2d = lm3_block(l2c, 128)

  # ENCODER - LAYER 3
  l3a = lm2b_block(l2d, 256)
  l3b = lm3_block(l3a, 256)
  l3c = lm3_block(l3b, 256)
  l3d = lm3_block(l3c, 256)
  l3e = lm3_block(l3d, 256)
  l3f = lm3_block(l3e, 256)

  # ENCODER - LAYER 4
  l4a = lm2b_block(l3f, 512)
  l4b = lm3_block(l4a, 512)
  l4c = lm3_block(l4b, 512)
  l4c = ZeroPadding2D(padding=((0,0), (0,1)))(l4c)

  # BASE LAYER
  # b1 = Conv2D(512, 3, strides=(2,2), padding='same')(l4c)
  b1 = BatchNormalization()(l4c)
  b1 = Activation('relu')(b1)

  # DECODER - LAYER 1 (paling bawah)
  l5a = lm4_block(b1, l4c, 256)
  l5b = lm5_block(l5a, 256)
  l5c = lm5_block(l5b, 256)

  # DECODER - LAYER 2
  #rszz = Lambda(lambda x: tf.image.resize(x, size=(44, 33), method=tf.image.ResizeMethod.BILINEAR))
  rszz = Lambda(resize_bilinear_44_33, output_shape=resize_output_shape_44_33)
  l6o = rszz(l5c)
  # pad6 = ZeroPadding2D(padding=((0, 0), (1, 2)))(l3f) # height (left, right), width (left, right)
  l6a = lm4_block(l6o, l3f, 128)
  l6b = lm5_block(l6a, 128)
  l6c = lm5_block(l6b, 128)

  # DECODER - LAYER 3 - di slice dan di zeropadding spy shape sama
  # Input
  #rzs = Lambda(lambda x: tf.image.resize(x, size=(87, 66), method=tf.image.ResizeMethod.BILINEAR))
  rzs = Lambda(resize_bilinear_87_66, output_shape=resize_output_shape_87_66)
  l7o = rzs(l6c)

  l7a = lm4_block(l7o, l2d, 64)
  l7b = lm5_block(l7a, 64)
  l7c = lm5_block(l7b, 64)

  # DECODER - LAYER 4
  #rzs2 = Lambda(lambda x: tf.image.resize(x, size=(174, 132), method=tf.image.ResizeMethod.BILINEAR))
  rzs2 = Lambda(resize_bilinear_174_132, output_shape=resize_output_shape_174_132)
  l8o = rzs2(l7c)

  l8a = lm4_block(l8o, l1d, 32)
  l8b = lm5_block(l8a, 32)
  l8c = lm5_block(l8b, 32)
  l8d = lm6_block(l8c, 32)
  #rzs4 = Lambda(lambda x: tf.image.resize(x, size=(682, 512), method=tf.image.ResizeMethod.BILINEAR))
  rzs4 = Lambda(resize_bilinear_682_512, output_shape=resize_output_shape_682_512)
  l8e = rzs4(l8d)

  # FINAL CONVOLUTION
  outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(l8e)  #Binary (can be multiclass)

  model = Model(inputs, outputs, name="U-Net")

  return model

def IoU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)

    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def dice_coef(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    sums = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)

    return (2. * intersection + 1.0) / (sums + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return 1-IoU(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function

def iou_metric(y_true, y_pred):
    return IoU(y_true, y_pred)

def loadmodel2(model_path):
    model = build_unbcsm(input_shape)
    # model.summary()

    sgd = SGD(momentum=0.9, nesterov=True)
    # model = load_model(model_path, compile=False)

    with open('model_architecture.json', 'w') as f:
      f.write(model.to_json())

    # Load model architecture
    with open('model_architecture.json', 'r') as f:
        model_json = f.read()

    custom_objects = {
        'resize_bilinear_44_33': resize_bilinear_44_33,
        'resize_bilinear_87_66': resize_bilinear_87_66,
        'resize_bilinear_174_132': resize_bilinear_174_132,
        'resize_bilinear_682_512': resize_bilinear_682_512,
        'resize_output_shape_44_33': resize_output_shape_44_33,
        'resize_output_shape_87_66': resize_output_shape_87_66,
        'resize_output_shape_174_132': resize_output_shape_174_132,
        'resize_output_shape_682_512': resize_output_shape_682_512
    }

    model = model_from_json(model_json, custom_objects=custom_objects)#, safe_mode=False)
    model.load_weights(model_path)
    model.compile(optimizer=sgd, loss='jacard_coef_loss', metrics=[iou_metric, dice_coef])

    return model

def read_image(model, image_path):
    SIZE_X = 512
    SIZE_Y = 682

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (SIZE_X, SIZE_Y))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    prediction = model.predict(np.expand_dims(image, axis=0))

    thresholded_prediction = (prediction > 0.5).astype(np.uint8)    # thresholded_prediction[0] HASIL SEGMENTASI GRAY
    thresholded_prediction2 = np.squeeze(thresholded_prediction, axis=0)

    mask2 = np.stack((thresholded_prediction2,)*3, axis=-1)
    mask2 = np.squeeze(mask2, axis=2)
    img = image/255
    blended1 = img*mask2    # HASIL SEGMENTASI
    blended2 = (blended1 * 255).astype(np.uint8)
    gray_img = cv2.cvtColor((blended1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    return image, gray_img, thresholded_prediction[0], blended2
