import os
import glob
import shutil
import numpy as np
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# set the pseudo-random generator to a constant, so the training results are comparable
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications import resnet_v2, inception_v3, vgg16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
INPUT_SHAPE = (224, 224, 3)

def main(args):
    print('DEBUG INFO')
    print('TF Version: {}'.format(tf.__version__))
    if tf.__version__ != "2.1.0":
        raise ValueError("TensorFlow version should be 2.1.0")
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        raise ValueError("Please install GPU version of TF")

    set_precision(args.precision)

    print('---')
    print('STARTING TRAINING AND VALIDATION')
    print('model: {}, batch size: {}, epochs: {}'.format(args.model, args.batch_size, args.epochs))
    print('precision: {}'.format(args.precision))

    print('---')
    print('Splitting data...')
    split_data(20000, 5000)
    
    print('Building generators...')
    train_generator, validation_generator = build_generators(args.batch_size)
    
    print('Building model {}...'.format(args.model))
    # weights=None, classes=None
    if args.model == 'vgg':
        base_model = vgg16.VGG16(include_top=False, input_shape=INPUT_SHAPE)
    elif args.model == 'inception':
        base_model = inception_v3.InceptionV3(include_top=False, input_shape=INPUT_SHAPE)
    elif args.model == 'resnet':
        base_model = resnet_v2.ResNet152V2(include_top=False, input_shape=INPUT_SHAPE)
    model = build_model(base_model)
    
    print('Training model {}...'.format(args.model))
    train(model, train_generator, validation_generator, args.epochs)


def set_precision(precision):
    if precision == 'float16':
        dtype='float16'
        K.set_floatx(dtype)
        # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
        K.set_epsilon(1e-4)
    elif precision == 'mixed':
        policy = mixed_precision.Policy('mixed_float16')
    else:
        policy = mixed_precision.Policy('float32')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)


def split_data(train_size, val_size):
  files = glob.glob('data/train/*')

  train_files = np.random.choice(files, size=train_size, replace=False)
  cat_train = [fn for fn in train_files if 'cat' in fn]
  dog_train = [fn for fn in train_files if 'dog' in fn]
  
  files = np.asarray(list(set(files) - set(train_files)))
  val_files = np.random.choice(files, size=val_size, replace=False)
  cat_val = [fn for fn in val_files if 'cat' in fn]
  dog_val = [fn for fn in val_files if 'dog' in fn]

  train_dir = 'training_data'
  val_dir = 'validation_data'

  os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
  os.mkdir(os.path.join(train_dir,'cat')) if not os.path.isdir(os.path.join(train_dir,'cat')) else None
  os.mkdir(os.path.join(train_dir,'dog')) if not os.path.isdir(os.path.join(train_dir,'dog')) else None
  os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
  os.mkdir(os.path.join(val_dir,'cat')) if not os.path.isdir(os.path.join(val_dir,'cat')) else None
  os.mkdir(os.path.join(val_dir,'dog')) if not os.path.isdir(os.path.join(val_dir,'dog')) else None

  for fn in cat_train:
      shutil.copy(fn, os.path.join(train_dir,'cat'))
  for fn in dog_train:
      shutil.copy(fn, os.path.join(train_dir,'dog'))

  for fn in cat_val:
      shutil.copy(fn, os.path.join(val_dir,'cat'))
  for fn in dog_val:
      shutil.copy(fn, os.path.join(val_dir,'dog'))


def build_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255)

    val_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'training_data',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = val_gen.flow_from_directory(
        'validation_data',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')
    
    return train_generator, validation_generator


def build_model(base_model, learn_rate=0.0001):
    for layer in base_model.layers:
      layer.trainable = False
    x = base_model.output
    x=Flatten()(x)
    x=Dense(128,activation='relu')(x)
    preds=Dense(1,activation='sigmoid')(x)
    model = Model(inputs=base_model.input,outputs=preds)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'])

    return model


def train(model, train_generator, validation_generator, epochs):
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)   

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', 
        help='Specify precision. float32 (default), float16 or mixed (for mixed precision)', 
        type=str, choices=['mixed', 'float16', 'float32'], default='float32')

    parser.add_argument('--batch_size',
        help='Specify batch_size, default 32.',
        type=int, choices=[16, 32, 64, 128, 256, 512], default=32)
    
    parser.add_argument('--epochs',
        help='Specify number of epochs, default 8',
        type=int, default=8)

    parser.add_argument('--model',
    help='Specify model type, vgg, inception or resnet, default is inception',
    type=str, choices=['vgg', 'resnet', 'inception'], default='inception')

    args = parser.parse_args()
    main(args)