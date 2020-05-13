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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow.keras.backend as K

from statsmodels.stats.contingency_tables import mcnemar

INPUT_SHAPE = (224, 224, 3)
DEBUG = True

def main(args):
    print_debug('TF Version: {}'.format(tf.__version__))
    if tf.__version__ != "2.2.0":
        raise ValueError("TensorFlow version should be 2.2.0")
    if tf.test.gpu_device_name():
        print_debug('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        raise ValueError("Please install GPU version of TF")

    print('STARTING RUN: {}'.format(args.run))
    print('model: {}, batch size: {}, epochs: {}'.format(args.model, args.batch_size, args.epochs))
    print('---')
    split_data(20000, 5000)

    print('Building generators...')
    train_generator = build_generator('training_data', args.batch_size)
    validation_generator = build_generator('validation_data', args.batch_size)

    float32_path = args.model + '-float32'
    mixed_path = args.model + '-mixed'
    if args.run == 'train':
        print('---')
        print('Training model {} with mixed precision...'.format(args.model))
        print('Building model {}...'.format(args.model))
        set_precision('mixed')

        if args.model == 'vgg':
            base_model = vgg16.VGG16(include_top=False, input_shape=INPUT_SHAPE)
        elif args.model == 'inception':
            base_model = inception_v3.InceptionV3(include_top=False, input_shape=INPUT_SHAPE)
        elif args.model == 'resnet':
            base_model = resnet_v2.ResNet152V2(include_top=False, input_shape=INPUT_SHAPE)
        model = build_model(base_model)
        model_mixed = train(model, train_generator, validation_generator, args.epochs, mixed_path)
        #model_mixed.save('testtemp')

        print('---')
        print('Training model {} with float32 precision...'.format(args.model))
        print('Building model {}...'.format(args.model))
        set_precision('float32')

        if args.model == 'vgg':
            base_model = vgg16.VGG16(include_top=False, input_shape=INPUT_SHAPE)
        elif args.model == 'inception':
            base_model = inception_v3.InceptionV3(include_top=False, input_shape=INPUT_SHAPE)
        elif args.model == 'resnet':
            base_model = resnet_v2.ResNet152V2(include_top=False, input_shape=INPUT_SHAPE)
        model = build_model(base_model)
        model_float32 = train(model, train_generator, validation_generator, args.epochs, float32_path)

    elif args.run == 'test':
        if os.path.exists(float32_path) and os.path.exists(mixed_path):
            print('loading pre-trained models')
            model_float32 = load_model(float32_path)
            model_mixed = load_model(mixed_path)
            test_generator = build_generator('validation_data', args.batch_size)
            mcnemar_test(model_float32, model_mixed, test_generator)
        else:
            raise ValueError('no models found')

    if False:
        print('running precision')
        trained_model.predict(x=validation_generator, workers=4, verbose=1)

        print('running precision 2')
        trained_model.predict(x=validation_generator, workers=4, verbose=1)


def set_precision(precision):
    if precision == 'float16':
        dtype='float16'
        K.set_floatx(dtype)
        # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
        K.set_epsilon(1e-4)
        print_debug('Compute dtype: %s' % 'float16')
        print_debug('Variable dtype: %s' % 'float16')
    elif precision == 'mixed':
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print_debug('Compute dtype: %s' % policy.compute_dtype)
        print_debug('Variable dtype: %s' % policy.variable_dtype)
    else:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_policy(policy)
        print_debug('Compute dtype: %s' % policy.compute_dtype)
        print_debug('Variable dtype: %s' % policy.variable_dtype)


def split_data(train_size, val_size):
    train_dir = 'training_data'
    val_dir = 'validation_data'

    train_files = sum([len(files) for r, d, files in os.walk(train_dir)])
    val_files =  sum([len(files) for r, d, files in os.walk(val_dir)])

    if train_files == train_size and val_files == val_size:
        print('Data is already split')
    else:
        print('Splitting data...')
        files = glob.glob('data/train/*')

        train_files = np.random.choice(files, size=train_size, replace=False)
        cat_train = [fn for fn in train_files if 'cat' in fn]
        dog_train = [fn for fn in train_files if 'dog' in fn]

        files = np.asarray(list(set(files) - set(train_files)))
        val_files = np.random.choice(files, size=val_size, replace=False)
        cat_val = [fn for fn in val_files if 'cat' in fn]
        dog_val = [fn for fn in val_files if 'dog' in fn]



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


def build_generator(directory, batch_size):
    base_generator = ImageDataGenerator(
        rescale=1./255)

    generator = base_generator.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')
    
    return generator


def build_model(base_model, learn_rate=0.0001):
    for layer in base_model.layers:
      layer.trainable = False
    x = base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    preds=Dense(1,activation='sigmoid')(x)
    model = Model(inputs=base_model.input,outputs=preds)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'])

    return model


def train(model, train_generator, validation_generator, epochs, filepath):
    model_checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                                save_weights_only=False,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True,
                                                verbose=1)
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, workers=4,
              callbacks=[model_checkpoint_callback], steps_per_epoch=4, validation_steps=2)
    #
    #model.load_weights(filepath)
    #model.save('temp')
    #
    # , callbacks=[model_checkpoint_callback])

    return model

def mcnemar_test(model1, model2, test_generator):
    print("running McNemar's Test")
    yesyes = 0
    yesno = 0
    noyes = 0
    nono = 0
    total = len(test_generator)
    print('{} items to evaluate'.format(total))
    index = 0
    while index <= test_generator.batch_index:

        item = test_generator.next()

        model1_results = model1.evaluate(item[0], item[1], verbose=0)[1]
        model2_results = model2.evaluate(item[0], item[1], verbose=0)[1]
        if model1_results == 1:
            if model2_results == 1:
                yesyes+=1
            elif model2_results == 0:
                yesno+=1
            else:
                print('unknown scenario model1: {}, model2: {}'.format(model1_results, model2_results))
        elif model1_results == 0:
            if model2_results == 1:
                noyes+=1
            elif model2_results == 0:
                nono+=1
            else:
                print('unknown scenario model1: {}, model2: {}'.format(model1_results, model2_results))
        else:
            print('unknown scenario model1: {}, model2: {}'.format(model1_results, model2_results))

        index += 1
        if index % 500 == 0:
            percentage = int(index / total * 100)
            print('{}% done. yes/yes: {}, yes/no: {}, no/yes: {}, no/no: {}'.format(percentage, yesyes, yesno,
                                                                                    noyes, nono))
        elif index % 100 == 0:
            percentage = int(index / total * 100)
            print('{}% done.'.format(percentage))

    print('yes/yes: {}, yes/no: {}, no/yes: {}, no/no: {}'.format(yesyes, yesno, noyes, nono))

    table = [[yesyes, yesno],[noyes,nono]]
    result = mcnemar(table, exact=False, correction=True)

    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')


def print_debug(message):
    if DEBUG:
        print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
        help='Specify batch_size, default 32.',
        type=int, choices=[16, 32, 64, 128, 256, 512], default=32)
    
    parser.add_argument('--epochs',
        help='Specify number of epochs, default 8',
        type=int, default=8)

    parser.add_argument('--model',
        help='Specify model type, vgg, inception or resnet, default is inception',
        type=str, choices=['vgg', 'resnet', 'inception'], default='inception')

    parser.add_argument('--run',
        help='Specify type of run, train or test, default is train',
        type=str, choices=['train', 'test'], default='train')

    args = parser.parse_args()
    main(args)