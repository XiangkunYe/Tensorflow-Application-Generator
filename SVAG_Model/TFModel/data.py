import tensorflow as tf
import os
import functools
import re
import tensorflow.contrib as tfcontrib
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split


class BottleneckDataPipeline(object):

    def __init__(self, bottlenecks_dir, labels_dir):
        self.bottlenecks_dir = bottlenecks_dir
        self.labels_dir = labels_dir
        self.labels_classes = 0
        # records label's name with the number from 0 to labels_classes-1
        self.label_name_val_dict = {}
        self.training_count = 0
        self.testing_count = 0
        self.validation_count = 0
        self.update_labels_info(labels_dir)

    def update_labels_info(self, labels_dir):
        """
        read label cfg file to get the relationship between label and file names
        :param labels_dir: the location of label files
        :return:
        """
        with open(labels_dir, 'r') as labels_file:
            is_first_line = True
            for line in labels_file:
                if is_first_line or line[0] == '\n':
                    is_first_line = False
                    continue
                filename = line.split(',')[0]
                label = line.split(',')[1]
                self.label_name_val_dict[filename] = int(label)
        self.labels_classes = len(self.label_name_val_dict.keys())

    def _process_bottleneck(self, filename, label):
        """
        the mapping function to handle the bottleneck files
        :param filename: bottleneck file name
        :param label: the label
        :return:
        """
        bottleneck_string = tf.read_file(filename)
        bottleneck_values = tf.string_split([bottleneck_string], ',').values
        bottleneck = tf.strings.to_number(bottleneck_values, out_type=tf.float32)
        bottleneck = tf.reshape(bottleneck, [2048])
        label = tf.one_hot(label, self.labels_classes, axis=0)
        return bottleneck, label

    def extract_filenames_for_classify(self, vali_percentage):
        """
        extract filenames
        :param vali_percentage: validation percentage
        :return: filenames
        """
        if not gfile.Exists(self.bottlenecks_dir):
            print("Image directory '" + self.bottlenecks_dir + "' not found.")
            return None
        result = {}
        sub_dirs = [x[0] for x in gfile.Walk(self.bottlenecks_dir)]
        # The root directory comes first, so skip it.
        is_root_dir = True
        extensions = ['txt']
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == self.bottlenecks_dir:
                continue
            print("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(self.bottlenecks_dir, dir_name, '*.' + extension)
                file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                print('No files found')
                continue

            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images, validation_images = train_test_split(file_list, test_size=vali_percentage)
            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'validation': validation_images}
        return result

    def get_bottleneck_dataset(self, filenames, labels, threads=5, batch_size=100, shuffle=True):
        """
        get bottleneck dataset
        :param filenames: filenames
        :param labels: labels
        :param threads: map threads
        :param batch_size: batch_size
        :param shuffle: whether shuffle or not
        :return: dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._process_bottleneck, num_parallel_calls=threads)
        if shuffle:
            dataset = dataset.shuffle(len(filenames))
        dataset = dataset.repeat().batch(batch_size)
        return dataset

    def get_input_bottleneck(self, vali_percentage):
        """
        get all filenames and lebels
        :param vali_percentage: validation percentage
        :return:
        """
        result = {'training': {'filenames': [], 'labels': []},
                  'validation': {'filenames': [], 'labels': []},
                  # 'testing': {'filenames': [], 'labels': []},
                  }
        filenames = self.extract_filenames_for_classify(vali_percentage)
        class_count = len(filenames.keys())
        if class_count == 0:
            print('No valid folders of images found at ' + self.bottlenecks_dir)
            return -1
        if class_count == 1:
            print('Only one valid folder of images found at ' + self.bottlenecks_dir +
                  ' - multiple classes are needed for classification.')
            return -1

        for cls in filenames.keys():
            for k in result.keys():
                result[k]['filenames'] += filenames[cls][k]
                result[k]['labels'] += [self.label_name_val_dict[cls] for _ in filenames[cls][k]]

        self.training_count = len(result["training"]["filenames"])
        self.validation_count = len(result["validation"]["filenames"])

        return result

    def get_input_bottleneck_dataset(self, vali_percentage, batch_size):
        result = {}
        image_label_result = self.get_input_bottleneck(vali_percentage)

        result["training"] = self.get_bottleneck_dataset(image_label_result["training"]["filenames"],
                                                         image_label_result["training"]["labels"],
                                                         batch_size=batch_size)
        result["validation"] = self.get_bottleneck_dataset(image_label_result["validation"]["filenames"],
                                                           image_label_result["validation"]["labels"],
                                                           batch_size=batch_size)
        return result


class ImageDataPipeline(object):

    def __init__(self, images_dir, labels_dir, image_size=None, image_channels=3, method="Classification"):
        self.images_dir = images_dir
        if image_size is None:
            self.image_size = [64, 64]
        else:
            self.image_size = image_size
        self.image_channels = image_channels
        self.is_seg = (method == "Segmentation")
        self.max_images = 10000000
        self.labels_dir = labels_dir
        self.labels_classes = 0
        # records label's name with the number from 0 to labels_classes-1
        self.label_name_val_dict = {}
        self.training_count = 0
        self.testing_count = 0
        self.validation_count = 0
        self.update_labels_info(labels_dir)

    def update_labels_info(self, labels_dir):
        """
        read label cfg file to get the relationship between label and file names
        :param labels_dir: the location of label files
        :return:
        """
        if not self.is_seg:
            with open(labels_dir, 'r') as labels_file:
                is_first_line = True
                for line in labels_file:
                    if is_first_line or line[0] == '\n':
                        is_first_line = False
                        continue
                    filename = line.split(',')[0]
                    label = line.split(',')[1]
                    self.label_name_val_dict[filename] = int(label)
            self.labels_classes = len(self.label_name_val_dict.keys())

    def _process_pathnames(self, filename, label):
        """
        process image filename and label.
        :param filename: image filename
        :param label: expected label
        :return: image matrix: height X width X channels, label vector
        """
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=self.image_channels)

        if self.is_seg:
            label_image_string = tf.read_file(label)
            label_image = tf.image.decode_gif(label_image_string)[0]
            label_image = label_image[:, :, 0]
            label = tf.expand_dims(label_image, axis=-1)
        else:
            label = tf.one_hot(label, self.labels_classes, axis=0)

        return image, label

    def _flip_image(self, horizontal_flip, image, label):
        """
        flip the image( data augmentation )
        :param horizontal_flip: whether flip horizontally or not
        :param image: input image data
        :return: flipped image data
        """
        if horizontal_flip:
            flip_prob = tf.random_uniform([], 0.0, 1.0)
            if self.is_seg:
                image, label = tf.cond(tf.less(flip_prob, 0.5),
                                       lambda: (tf.image.flip_left_right(image), tf.image.flip_left_right(label)),
                                       lambda: (image, label))
            else:
                image = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: tf.image.flip_left_right(image),
                                lambda: image)
        return image, label

    def _shift_image(self, image, label, width_shift_range, height_shift_range):
        """
        shift image( data augmentation )
        :param image: input image data
        :param width_shift_range: shift range of image's width
        :param height_shift_range: shift range of images's height
        :return: shifted image data
        """
        if width_shift_range or height_shift_range:
            if width_shift_range:
                width_shift_range = tf.random_uniform([],
                                                      -width_shift_range * self.image_size[1],
                                                      width_shift_range * self.image_size[1])
            if height_shift_range:
                height_shift_range = tf.random_uniform([],
                                                       -height_shift_range * self.image_size[0],
                                                       height_shift_range * self.image_size[0])
            # Translate both
            image = tfcontrib.image.translate(image, [width_shift_range, height_shift_range])
            if self.is_seg:
                label = tfcontrib.image.translate(label, [width_shift_range, height_shift_range])

        return image, label

    def _image_label_preprocess(self,
                                image,
                                label,
                                resize=None,
                                scale=1,
                                hue_delta=0,
                                horizontal_flip=False,
                                width_shift_range=0,
                                height_shift_range=0):
        """
        image preprocess function
        :param image: image data
        :param resize: resize the image to some size, param resize is expected to be [width, height]
        :param scale: scale image, e.g. scale=1, then all the value will multiply 1 / 255.0
        :param hue_delta: adjust the hue of an RGB image by random factor
        :param horizontal_flip: whether flip the image horizontally
        :param width_shift_range: shift range of image's width
        :param height_shift_range: shift range of image's height
        :return:
        """
        if resize is not None:
            image = tf.image.resize_images(image, resize)
        if hue_delta:
            image = tf.image.random_hue(image, hue_delta)
        if self.is_seg and resize is not None:
            label = tf.image.resize_images(label, resize)
        image, label = self._flip_image(horizontal_flip, image, label)
        image, label = self._shift_image(image, label, width_shift_range, height_shift_range)
        if self.is_seg:
            label = tf.to_float(label) * scale
        image = tf.to_float(image) * scale
        return image, label

    def get_baseline_dataset(self,
                             filenames,
                             labels,
                             parse_fn,
                             threads=5,
                             batch_size=100,
                             shuffle=True):
        """
        get the basic dataset based on the image filenames and all labels
        :param filenames: image filenames
        :param labels: a list containing all the label val or the label images filenames
        :param parse_fn: the function to preprocess all the image and label data
        :param threads: taking advantage of multithreading
        :param batch_size: the batch size
        :param shuffle: whether shuffle the dataset or not
        :return:
        """
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._process_pathnames, num_parallel_calls=threads)
        dataset = dataset.map(parse_fn, num_parallel_calls=threads)
        if shuffle:
            dataset = dataset.shuffle(len(filenames))
        dataset = dataset.repeat().batch(batch_size)

        return dataset

    def extract_filenames_for_classify(self, vali_percentage):
        if not gfile.Exists(self.images_dir):
            print("Image directory '" + self.images_dir + "' not found.")
            return None
        result = {}
        sub_dirs = [x[0] for x in gfile.Walk(self.images_dir)]
        # The root directory comes first, so skip it.
        is_root_dir = True
        extensions = ['jpg', 'jpeg', 'png']
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == self.images_dir:
                continue
            print("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(self.images_dir, dir_name, '*.' + extension)
                file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                print('No files found')
                continue
            if len(file_list) < 20:
                print('WARNING: Folder has less than 20 images, which may cause issues.')
            elif len(file_list) > self.max_images:
                print('WARNING: Folder {} has more than {} images. Some images will '
                      'never be selected.'.format(dir_name, self.max_images))

            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images, validation_images = train_test_split(file_list, test_size=vali_percentage)
            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'validation': validation_images}
        return result

    def extract_images_filenames_for_segmentation(self, images_dir, vali_precentage):
        if not gfile.Exists(images_dir):
            print("Image directory '" + images_dir + "' not found.")
            return None
        result = {}
        extensions = ['jpg', 'jpeg', 'png', 'gif']
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(images_dir, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            return None
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > self.max_images:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(images_dir, self.max_images))
        training_images, validation_images = train_test_split(file_list, test_size=vali_precentage)
        result['segmentation'] = {
            'dir': images_dir,
            'training': training_images,
            'validation': validation_images}

        return result

    def get_input_files(self, vali_percentage):
        result = {'training': {'filenames': [], 'labels': []},
                  'validation': {'filenames': [], 'labels': []},
                  # 'testing': {'filenames': [], 'labels': []},
                  }
        if self.is_seg:
            images_filenames = self.extract_images_filenames_for_segmentation(self.images_dir, vali_percentage)
            for k in result.keys():
                result[k]["filenames"] = images_filenames["segmentation"][k]
                for filename in result[k]["filenames"]:
                    label_filename = os.path.basename(filename)
                    label_filename = '{}_mask.gif'.format(os.path.basename(label_filename).split('.')[0])
                    result[k]["labels"].append(os.path.join(self.labels_dir, label_filename))
        else:
            images_filenames = self.extract_filenames_for_classify(vali_percentage)
            class_count = len(images_filenames.keys())
            if class_count == 0:
                print('No valid folders of images found at ' + self.images_dir)
                return -1
            if class_count == 1:
                print('Only one valid folder of images found at ' + self.images_dir +
                      ' - multiple classes are needed for classification.')
                return -1

            for cls in images_filenames.keys():
                for k in result.keys():
                    result[k]['filenames'] += images_filenames[cls][k]
                    result[k]['labels'] += [self.label_name_val_dict[cls] for _ in images_filenames[cls][k]]

        self.training_count = len(result["training"]["filenames"])
        self.validation_count = len(result["validation"]["filenames"])

        return result

    def get_input_dataset(self, vali_percentage, batch_size):
        result = {}
        image_label_result = self.get_input_files(vali_percentage)

        if self.is_seg:
            train_cfg = {
                'resize': [self.image_size[0], self.image_size[1]],
                'scale': 1 / 255.,
                'hue_delta': 0.1,
                'horizontal_flip': True,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1
            }
            vali_cfg = {
                'resize': [self.image_size[0], self.image_size[1]],
                'scale': 1 / 255.
            }
            train_processing_fn = functools.partial(self._image_label_preprocess, **train_cfg)
            vali_processing_fn = functools.partial(self._image_label_preprocess, **vali_cfg)
            result["training"] = self.get_baseline_dataset(image_label_result["training"]["filenames"],
                                                           image_label_result["training"]["labels"],
                                                           parse_fn=train_processing_fn,
                                                           batch_size=batch_size)
            result["validation"] = self.get_baseline_dataset(image_label_result["validation"]["filenames"],
                                                             image_label_result["validation"]["labels"],
                                                             parse_fn=vali_processing_fn,
                                                             batch_size=batch_size)
        else:
            cfg = {
                'resize': [self.image_size[0], self.image_size[1]],
                'scale': 1 / 255.,
                'hue_delta': 0.1,
                'horizontal_flip': True,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1
            }
            image_processing_fn = functools.partial(self._image_label_preprocess, **cfg)
            result["training"] = self.get_baseline_dataset(image_label_result["training"]["filenames"],
                                                           image_label_result["training"]["labels"],
                                                           parse_fn=image_processing_fn,
                                                           batch_size=batch_size)
            result["validation"] = self.get_baseline_dataset(image_label_result["validation"]["filenames"],
                                                             image_label_result["validation"]["labels"],
                                                             parse_fn=image_processing_fn,
                                                             batch_size=batch_size)
        return result
