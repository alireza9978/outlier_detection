import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 5
IMG_SHAPE = [IMG_WIDTH, IMG_HEIGHT, 3]

# load data
data_dir = [('/home/alireza/projects/python/outlier/outlier_detection/datasets/flickr', "flickr"),
            ('/home/alireza/projects/python/outlier/outlier_detection/datasets/instagram', "instagram")]
output_dir = './extracted_feature/'
extractor_images_output_dir = './feature_extractor_images/'


def beautify(all_array, train_data_gen):
    column_to_be_added = np.array(train_data_gen.filenames)
    get_name = lambda t: str(t).split("/")[-1]
    column_to_be_added = np.array([get_name(xi) for xi in column_to_be_added])
    result = np.hstack((np.atleast_2d(column_to_be_added).T, all_array))
    header = "img"
    for i in range(result.shape[1] - 1):
        header = header + "," + str(i)
    data_format = ["%s"] + ["%s" for i in range(result.shape[1] - 1)]
    return result, header, data_format


def sample_step(input_image, model):
    disc_real_output = model(input_image)
    return disc_real_output.numpy()


def load_model_make_csv_50v2(train_data_gen, name):
    input_one = layers.Input(shape=IMG_SHAPE, name="input_image_layer")
    model = tf.keras.applications.ResNet50V2(include_top=True,
                                             weights='imagenet',
                                             input_tensor=input_one)

    model = tf.keras.Model(inputs=input_one, outputs=model.layers[len(model.layers) - 2].output)
    tf.keras.utils.plot_model(
        model, to_file=extractor_images_output_dir + 'ResNet50V2.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )

    image_batch, _ = train_data_gen.next()
    all_array = sample_step(image_batch, model)
    for i in range(len(train_data_gen) - 1):
        image_batch, _ = train_data_gen.next()
        all_array = np.concatenate((sample_step(image_batch, model), all_array), axis=0)

    all_array, header, data_format = beautify(all_array, train_data_gen)
    np.savetxt(output_dir + name + '_ResNet50V2.csv', all_array, fmt=data_format, header=header, delimiter=",")


def load_model_make_csv_101v2(train_data_gen, name):
    input_one = layers.Input(shape=IMG_SHAPE, name="input_image_layer")
    model = tf.keras.applications.ResNet101V2(include_top=True,
                                              weights='imagenet',
                                              input_tensor=input_one)

    model = tf.keras.Model(inputs=input_one, outputs=model.layers[len(model.layers) - 2].output)
    tf.keras.utils.plot_model(
        model, to_file=extractor_images_output_dir + 'ResNet101V2.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )

    image_batch, _ = train_data_gen.next()
    all_array = sample_step(image_batch, model)
    for i in range(len(train_data_gen) - 1):
        image_batch, _ = train_data_gen.next()
        all_array = np.concatenate((sample_step(image_batch, model), all_array), axis=0)

    all_array, header, data_format = beautify(all_array, train_data_gen)
    np.savetxt(output_dir + name + '_ResNet101V2.csv', all_array, fmt=data_format, header=header, delimiter=",")


def load_model_make_csv_152v2(train_data_gen, name):
    input_one = layers.Input(shape=IMG_SHAPE, name="input_image_layer")
    model = tf.keras.applications.ResNet152V2(include_top=True,
                                              weights='imagenet',
                                              input_tensor=input_one)

    model = tf.keras.Model(inputs=input_one, outputs=model.layers[len(model.layers) - 2].output)
    tf.keras.utils.plot_model(
        model, to_file=extractor_images_output_dir + 'ResNet152V2.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )

    image_batch, _ = train_data_gen.next()
    all_array = sample_step(image_batch, model)
    for i in range(len(train_data_gen) - 1):
        image_batch, _ = train_data_gen.next()
        all_array = np.concatenate((sample_step(image_batch, model), all_array), axis=0)

    all_array, header, data_format = beautify(all_array, train_data_gen)
    np.savetxt(output_dir + name + '_ResNet152V2.csv', all_array, fmt=data_format, header=header, delimiter=",")


def load_model_make_csv_121(train_data_gen, name):
    input_one = layers.Input(shape=IMG_SHAPE, name="input_image_layer")
    model = tf.keras.applications.DenseNet121(include_top=True,
                                              weights='imagenet',
                                              input_tensor=input_one)

    model = tf.keras.Model(inputs=input_one, outputs=model.layers[len(model.layers) - 2].output)
    tf.keras.utils.plot_model(
        model, to_file=extractor_images_output_dir + 'DenseNet121.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )

    image_batch, _ = train_data_gen.next()
    all_array = sample_step(image_batch, model)
    for i in range(len(train_data_gen) - 1):
        image_batch, _ = train_data_gen.next()
        all_array = np.concatenate((sample_step(image_batch, model), all_array), axis=0)

    all_array, header, data_format = beautify(all_array, train_data_gen)
    np.savetxt(output_dir + name + '_DenseNet121.csv', all_array, fmt=data_format, header=header, delimiter=",")


def load_model_make_csv_169(train_data_gen, name):
    input_one = layers.Input(shape=IMG_SHAPE, name="input_image_layer")
    model = tf.keras.applications.DenseNet169(include_top=True,
                                              weights='imagenet',
                                              input_tensor=input_one)

    model = tf.keras.Model(inputs=input_one, outputs=model.layers[len(model.layers) - 2].output)
    tf.keras.utils.plot_model(
        model, to_file=extractor_images_output_dir + 'DenseNet169.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )

    image_batch, _ = train_data_gen.next()
    all_array = sample_step(image_batch, model)
    for i in range(len(train_data_gen) - 1):
        image_batch, _ = train_data_gen.next()
        all_array = np.concatenate((sample_step(image_batch, model), all_array), axis=0)

    all_array, header, data_format = beautify(all_array, train_data_gen)
    np.savetxt(output_dir + name + '_DenseNet169.csv', all_array, fmt=data_format, header=header, delimiter=",")


def load_model_make_csv_201(train_data_gen, name):
    input_one = layers.Input(shape=IMG_SHAPE, name="input_image_layer")
    model = tf.keras.applications.DenseNet201(include_top=True,
                                              weights='imagenet',
                                              input_tensor=input_one)

    model = tf.keras.Model(inputs=input_one, outputs=model.layers[len(model.layers) - 2].output)
    tf.keras.utils.plot_model(
        model, to_file=extractor_images_output_dir + 'DenseNet201.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )

    image_batch, _ = train_data_gen.next()
    all_array = sample_step(image_batch, model)
    for i in range(len(train_data_gen) - 1):
        image_batch, _ = train_data_gen.next()
        all_array = np.concatenate((sample_step(image_batch, model), all_array), axis=0)

    all_array, header, data_format = beautify(all_array, train_data_gen)
    np.savetxt(output_dir + name + '_DenseNet201.csv', all_array, fmt=data_format, header=header, delimiter=",")


def load_all_models(path, name):
    data_dir = pathlib.Path(path)
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=False,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH))
    load_model_make_csv_121(train_data_gen, name)
    load_model_make_csv_169(train_data_gen, name)
    load_model_make_csv_201(train_data_gen, name)
    load_model_make_csv_50v2(train_data_gen, name)
    load_model_make_csv_101v2(train_data_gen, name)
    load_model_make_csv_152v2(train_data_gen, name)


for path in data_dir:
    load_all_models(path[0], path[1])
