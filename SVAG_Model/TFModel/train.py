import TFModel.models
import TFModel.data
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from data import ImageDataPipeline, BottleneckDataPipeline
from models import InceptionModelGenerator
import os

IMAGE_SIZE = [256, 256]
CHANNELS = 3
SEGMENTATION_IMAGE_BATCH_SIZE = 100.
CLASSIFY_IMAGE_BATCH_SIZE = 10.
BOTTLENECK_BATCH_SIZE = 500.
VALIDATION_PERCENTAGE = 0.3
CLASSIFY_LEARNING_RATE = 0.0005
CLASSIFY_EPOCH = 20

IMAGE_FOLDER = 'images'
LABEL_FOLDER = 'labels'
LABEL_INFO_FILE = 'labels_info.txt'
BOTTLENECK_FOLDER = 'bottlenecks'
TRAINED_MODEL_FOLDER = 'models'
INCEPTION_V3_FILE = r'..\pre_train_file\inceptionV3_bottleneck.hdf5'


def get_and_cache_bottlenecks(file_dir, model_gen, image_dp):
    bottleneck_dir = os.path.join(file_dir, BOTTLENECK_FOLDER)

    image_ds = image_dp.get_input_files(VALIDATION_PERCENTAGE)
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
        if i % 100 == 0:
            print("already output {} images' bottlenecks")


def train_one_layer_model(file_dir, model_gen, bottleneck_dp):
    model_save_path = os.path.join(file_dir, TRAINED_MODEL_FOLDER, 'one_layer_weights.hdf5')

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
    return model_save_path


def train_classification(file_dir):
    saved_models_path = os.path.join(file_dir, TRAINED_MODEL_FOLDER)
    images_dir = os.path.join(file_dir, IMAGE_FOLDER)
    label_file = os.path.join(file_dir, LABEL_INFO_FILE)

    image_dp = ImageDataPipeline(images_dir, label_file, image_size=IMAGE_SIZE)
    generator = InceptionModelGenerator(INCEPTION_V3_FILE,
                                        image_dp.labels_classes,
                                        (IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS))
    bottleneck_dp = BottleneckDataPipeline(images_dir, label_file)

    get_and_cache_bottlenecks(file_dir, generator, image_dp)
    weights_file = train_one_layer_model(file_dir, generator, bottleneck_dp)
    saved_models_path = os.path.join(saved_models_path, weights_file)

    model = generator.get_eval_model()
    model.load_weights(saved_models_path, by_name=True)
    # TO DO: save model...


def train_segmentation(file_dir):
    pass


train_func = {'Classification': train_classification,
              'Segmentation': train_segmentation,
             }

def train(type, file_dir):
    train_func[type](file_dir)
