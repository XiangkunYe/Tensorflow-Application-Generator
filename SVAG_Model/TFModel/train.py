"""
The train.py implement how to train a model for image classification or segmentation purposes.
"""
import numpy as np
import tensorflow as tf
import os
import logging
import shutil
from tensorflow.python.platform import gfile
from data import ImageDataPipeline, BottleneckDataPipeline
from models import InceptionModelGenerator, UnetModelGenerator, MobileNetModelGenerator
from thread import TaskManager
from save_model import save_model
from build_app import build_android_app


IMAGE_SIZE = [224, 224]
CHANNELS = 3
SEGMENTATION_IMAGE_BATCH_SIZE = 5
CLASSIFY_IMAGE_BATCH_SIZE = 10
BOTTLENECK_BATCH_SIZE = 500
VALIDATION_PERCENTAGE = 0.3
CLASSIFY_LEARNING_RATE = 0.0005
CLASSIFY_EPOCH = 20
SEGMENTATION_LEARNING_RATE = 0.001
SEGMENTATION_EPOCH = 5

IMAGE_FOLDER = 'images'
LABEL_FOLDER = 'labels'
LABEL_INFO_FILE = 'labels_info.txt'
BOTTLENECK_FOLDER = 'bottlenecks'
TRAINED_MODEL_FOLDER = 'models'
INCEPTION_V3_FILE = r'.\pre_train_file\inceptionV3.hdf5'
MOBILENET_V1_FILE = r'.\pre_train_file\mobileNetV1.hdf5'
SAVE_ONE_LAYER_FILE = 'one_layer_weights.hdf5'
SAVE_FINAL_MODEL_FILE = 'Classification_model.hdf5'
SAVE_FINAL_MODEL_PB_FILE = 'Classification_model.pb'
SAVE_SEGMENTATION_WEIGHTS_FILE = 'Segmentation_weights.hdf5'
SAVE_SEGMENTATION_MODEL_FILE = 'Segmentation_model.hdf5'

TRAIN_LOGGER = logging.getLogger('task_manager')


def update_progress(task_id, progress):
    TaskManager().task_dict[task_id]['progress'] = progress


def get_and_cache_bottlenecks(task_id, file_dir, model_gen, image_dp):
    """
    generate bottleneck for every images and save them
    :param task_id: task id
    :param file_dir: where the images are saved
    :param model_gen: model generater, get the inception v3 model
    :param image_dp: image data pipeline
    :return:
    """
    TRAIN_LOGGER.info("[task({})]start generating bottlenecks".format(task_id))
    bottleneck_dir = os.path.join(file_dir, BOTTLENECK_FOLDER)

    image_ds = image_dp.get_input_files(VALIDATION_PERCENTAGE)
    if image_ds is None:
        return False
    images = image_ds["training"]["filenames"] + image_ds["validation"]["filenames"]
    labels = image_ds["training"]["labels"] + image_ds["validation"]["labels"]

    bottleneck_model = model_gen.get_bottleneck_model()

    label_class_dict = {}
    for cls in image_dp.label_name_val_dict.keys():
        label_class_dict[image_dp.label_name_val_dict[cls]] = cls
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        if not gfile.Exists(os.path.join(bottleneck_dir, label_class_dict[label])):
            os.mkdir(os.path.join(bottleneck_dir, label_class_dict[label]))
        bottleneck_file_name = os.path.basename(img).split('.')[0] + ".txt"

        bottleneck_path = os.path.join(bottleneck_dir, label_class_dict[label], bottleneck_file_name)

        img_string = tf.read_file(img)
        image = tf.image.decode_jpeg(img_string, channels=3)
        image = tf.image.resize_images(image, IMAGE_SIZE)
        image = tf.to_float(image) * (1 / 255.)
        image_data = tf.keras.backend.get_session().run(image)
        image_data = np.reshape(image_data, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS))

        bottleneck_value = np.squeeze(bottleneck_model.predict(image_data))
        bottleneck_string = ','.join(str(x) for x in bottleneck_value)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
        update_progress(task_id, ((i+0.) / len(images)) * 70.)
        if i % 100 == 0:
            TRAIN_LOGGER.info("[task({})]already output {} images' bottlenecks".format(task_id, i))
    TRAIN_LOGGER.info("[task({})]bottlenecks generation is done".format(task_id))
    return True


# def get_and_cache_bottlenecksV2(task_id, file_dir, model_gen, image_dp):
#     """
#         generate bottleneck for every images and save them
#         :param task_id: task id
#         :param file_dir: where the images are saved
#         :param model_gen: model generater, get the inception v3 model
#         :param image_dp: image data pipeline
#         :return:
#         """
#     TRAIN_LOGGER.info("[task({})]start generating bottlenecks".format(task_id))
#     bottleneck_dir = os.path.join(file_dir, BOTTLENECK_FOLDER)
#
#     image_ds = image_dp.get_input_files(VALIDATION_PERCENTAGE)
#     if image_ds is None:
#         return False
#     images = image_ds["training"]["filenames"] + image_ds["validation"]["filenames"]
#     labels = image_ds["training"]["labels"] + image_ds["validation"]["labels"]
#
#     bottleneck_model = model_gen.get_bottleneck_model()
#
#     label_class_dict = {}
#     for cls in image_dp.label_name_val_dict.keys():
#         label_class_dict[image_dp.label_name_val_dict[cls]] = cls
#
#     # create dataset
#     dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#     dataset = dataset.map()



def train_one_layer_model(task_id, file_dir, model_gen, bottleneck_dp):
    """
    train the final layer for image classification
    :param task_id: task id
    :param file_dir: where images are saved
    :param model_gen: model generator, get the final layer model
    :param bottleneck_dp: bottleneck data pipeline
    :return:
    """
    TRAIN_LOGGER.info("[task({})]start training the final layer model".format(task_id))
    model_save_path = os.path.join(file_dir, TRAINED_MODEL_FOLDER, SAVE_ONE_LAYER_FILE)

    bottleneck_ds = bottleneck_dp.get_input_bottleneck_dataset(VALIDATION_PERCENTAGE, BOTTLENECK_BATCH_SIZE)
    one_layer_model = model_gen.get_train_model()
    self_adam = model_gen.get_optimizer(learning_rate=CLASSIFY_LEARNING_RATE)
    one_layer_model.compile(optimizer=self_adam, loss='categorical_crossentropy', metrics=['accuracy'])
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                       monitor='val_acc', save_best_only=True, verbose=1)

    one_layer_model.fit(bottleneck_ds["training"],
                        steps_per_epoch=int(np.ceil(bottleneck_dp.training_count / BOTTLENECK_BATCH_SIZE)),
                        epochs=CLASSIFY_EPOCH,
                        validation_data=bottleneck_ds["validation"],
                        validation_steps=int(np.ceil(bottleneck_dp.validation_count / BOTTLENECK_BATCH_SIZE)),
                        callbacks=[save_callback])
    update_progress(task_id, 90)
    TRAIN_LOGGER.info("[task({})]finish training".format(task_id))


def train_segmentation_model(task_id, file_dir, model_gen, datapipeline):
    """
    train the image segmentation model based on model_gen
    :param task_id: task id
    :param file_dir: where images are saved
    :param model_gen: model generator
    :param datapipeline: image datapipeline
    :return:
    """
    TRAIN_LOGGER.info("[task({})]start training".format(task_id))
    saved_models_path = os.path.join(file_dir, TRAINED_MODEL_FOLDER, SAVE_SEGMENTATION_WEIGHTS_FILE)
    image_ds = datapipeline.get_input_dataset(VALIDATION_PERCENTAGE, SEGMENTATION_IMAGE_BATCH_SIZE)
    if image_ds is None:
        return False

    train_model = model_gen.get_model()
    self_adam = model_gen.get_optimizer(learning_rate=SEGMENTATION_LEARNING_RATE)
    train_model.compile(optimizer=self_adam,
                        loss=model_gen.bce_dice_loss,
                        metrics=[model_gen.dice_loss])
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saved_models_path,
                                                       monitor='val_dice_loss', save_best_only=True, verbose=1)
    train_model.fit(image_ds['training'],
                    steps_per_epoch=int(np.ceil(datapipeline.training_count / SEGMENTATION_IMAGE_BATCH_SIZE)),
                    epochs=SEGMENTATION_EPOCH,
                    validation_data=image_ds["validation"],
                    validation_steps=int(np.ceil(datapipeline.validation_count / SEGMENTATION_IMAGE_BATCH_SIZE)),
                    callbacks=[save_callback])
    update_progress(task_id, 90)
    TRAIN_LOGGER.info("[task({})]finish training".format(task_id))


def train_classification(task_id, file_dir):
    """
    train model for image classification
    :param task_id: task id
    :param file_dir: where images are saved
    :return:
    """
    # check the file_dir
    if not os.path.exists(file_dir):
        TRAIN_LOGGER.info("{} not exists. training task stop".format(file_dir))
        return None

    saved_models_path = os.path.join(file_dir, TRAINED_MODEL_FOLDER)
    images_dir = os.path.join(file_dir, IMAGE_FOLDER)
    label_file = os.path.join(file_dir, LABEL_INFO_FILE)
    bottleneck = os.path.join(file_dir, BOTTLENECK_FOLDER)
    weights_file = os.path.join(saved_models_path, SAVE_ONE_LAYER_FILE)
    model_file = os.path.join(saved_models_path, SAVE_FINAL_MODEL_FILE)

    # check or create file dir needed
    if not os.path.exists(images_dir):
        TRAIN_LOGGER.info("{} not exists. training task stop".format(images_dir))
        return None
    if not os.path.exists(label_file):
        TRAIN_LOGGER.info("{} not exists. training task stop".format(label_file))
        return None
    if os.path.exists(bottleneck):
        shutil.rmtree(bottleneck)
    os.mkdir(bottleneck)
    if not os.path.exists(saved_models_path):
        os.mkdir(saved_models_path)

    image_dp = ImageDataPipeline(images_dir, label_file, image_size=IMAGE_SIZE)
    generator = InceptionModelGenerator(INCEPTION_V3_FILE,
                                        image_dp.labels_classes,
                                        (IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS))
    bottleneck_dp = BottleneckDataPipeline(bottleneck, label_file)

    # Step 1. compute and save bottlenecks
    update_progress(task_id, 0)
    succeed = get_and_cache_bottlenecks(task_id, file_dir, generator, image_dp)
    if not succeed:
        TRAIN_LOGGER.info("[task({})]get and cache bottlenecks failed".format(task_id))
        return None
    # Step 2. train the last layer of our model
    update_progress(task_id, 70)
    train_one_layer_model(task_id, file_dir, generator, bottleneck_dp)
    # Step 3. load the weights generated in Step 2. to our final model
    model = generator.get_eval_model()
    model.load_weights(weights_file, by_name=True)
    # TO DO: save model as android format...
    model.save(model_file)
    update_progress(task_id, 90)
    TRAIN_LOGGER.info("[task({})]model saved".format(task_id))
    # Step 4. Clear all the bottlenecks
    shutil.rmtree(bottleneck)
    return model_file


def train_segmentation(task_id, file_dir):
    """
    train a model for image segmentation
    :param task_id: task id
    :param file_dir: where images are saved
    :return:
    """
    saved_models_path = os.path.join(file_dir, TRAINED_MODEL_FOLDER)
    images_dir = os.path.join(file_dir, IMAGE_FOLDER)
    label_dir = os.path.join(file_dir, LABEL_FOLDER)
    model_file = os.path.join(saved_models_path, SAVE_SEGMENTATION_MODEL_FILE)

    if not os.path.exists(saved_models_path):
        os.mkdir(saved_models_path)
    if not os.path.exists(images_dir):
        TRAIN_LOGGER.info("{} not exists. training task stop".format(images_dir))
        return None
    if not os.path.exists(label_dir):
        TRAIN_LOGGER.info("{} not exists. training task stop".format(label_dir))
        return None

    image_dp = ImageDataPipeline(images_dir, label_dir, image_size=IMAGE_SIZE, method="Segmentation")
    generator = UnetModelGenerator((IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS))

    # start training...
    update_progress(task_id, 0)
    succeed = train_segmentation_model(task_id, file_dir, generator, image_dp)
    if not succeed:
        TRAIN_LOGGER.info("[task({})]train segmentation model failed".format(task_id))
        return None

    model = generator.get_model()
    model.save(model_file)
    update_progress(task_id, 100)
    TRAIN_LOGGER.info("[task({})]model saved".format(task_id))
    return model_file


train_func = {'classification': train_classification,
              'segmentation': train_segmentation,
              }


def train(task_id, method, file_dir):
    # support multi thread, each thread create a new session with new graph
    with tf.Session(graph=tf.Graph()) as sess:
        tf.keras.backend.set_session(sess)
        train_fc = train_func.get(method.lower(), None)
        if train_fc is None:
            TRAIN_LOGGER.info("[task{}]invalid method. not classification or segmentation".format(task_id))
            return None
        model_file = train_fc(task_id, file_dir)
    # save as .pb file
    if model_file is not None:
        saved_models_path = os.path.join(file_dir, TRAINED_MODEL_FOLDER)
        save_model(saved_models_path, SAVE_FINAL_MODEL_PB_FILE, model_file)

        # build android app
        images_dir = os.path.join(file_dir, IMAGE_FOLDER)
        label_file = os.path.join(file_dir, LABEL_INFO_FILE)
        image_dp = ImageDataPipeline(images_dir, label_file, image_size=IMAGE_SIZE)
        labels = ['' for _ in range(image_dp.labels_classes)]
        for key in image_dp.label_name_val_dict.keys():
            labels[image_dp.label_name_val_dict[key]] = key
        is_success, outputs = build_android_app(labels, saved_models_path, model_file)
        if not is_success:
            return None
        return outputs

    return model_file
