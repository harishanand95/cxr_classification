import os
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
from six.moves import cPickle as pickle
from sklearn.utils import shuffle


class ChinaCXRDataset:
    """Dataset model to fetch images from NLM-ChinaCXRSet"""

    def __init__(self, folder):
        self._folder = folder
        self._image_files = os.listdir(self._folder)
        self._valid_images_count = 0
        self._num_of_files = len(self._image_files)
        self._image_width = 0
        self._image_height = 0
        self._convert_to_gray = True
        self._dataset = None
        self._labels = None
        self._dataset_filename = "CXR_png.pickle"
        self._test_data_size = 0
        self._test_dataset = None
        self._test_labels = None

    def load_images(self, image_width, image_height, pixel_depth, convert_to_gray=True):
        self._image_width = image_width
        self._image_height = image_height
        self._convert_to_gray = convert_to_gray
        if convert_to_gray is True:
            self._dataset = np.ndarray(shape=(self._num_of_files, image_width, image_height, 1),
                                       dtype=np.float32)
        else:
            self._dataset = np.ndarray(shape=(self._num_of_files, image_width, image_height, 3),
                                       dtype=np.float32)  # RGB
        self._labels = np.ndarray(shape=(self._num_of_files), dtype=np.int32)
        num_images = 0
        for image in self._image_files:
            try:
                image_file = os.path.join(self._folder, image)
                img = mpimg.imread(image_file)
                if convert_to_gray is True:
                    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                img_scaled = misc.imresize(img, (image_width, image_height))
                # Check if this creates any problems when color image is passed.
                image_data = (img_scaled.astype(float) - pixel_depth / 2) / pixel_depth
                image_data = image_data.reshape((image_width, image_height, 1 if self._convert_to_gray is True else 3))
                if image_data.shape != (image_width, image_height, 1 if self._convert_to_gray is True else 3):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                self._dataset[num_images, :, :, :] = image_data
                # If filename ends with a 1, it means the image is a case of TB reported.
                if str(image_file[-5]) == "1":
                    self._labels[num_images] = 1
                else:
                    self._labels[num_images] = 0
                num_images += 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        # Limits dataset to only valid images found

        self._valid_images_count = num_images
        self._dataset = self._dataset[0:num_images, :, :, :]
        self._labels = self._labels[0:num_images]
        shuffle(self._dataset, self._labels, random_state=0)
        print('Full dataset tensor shape :', self._dataset.shape)

    def load_from_pickle(self, dataset_filename="CXR_png.pickle"):
        if not os.path.isfile(dataset_filename):
            print('No file named ', dataset_filename, ' exists')
            return
        else:
            self._dataset_filename = dataset_filename
            with open(self._dataset_filename, 'rb') as f:
                data = pickle.load(f)
                self._dataset = data["dataset"]
                self._labels = data["labels"]
                self._image_height = data["height"]
                self._image_width = data["width"]
                self._valid_images_count = data["valid_images_count"]
                self._convert_to_gray = data["convert_to_gray"]
                self._folder = data["folder"]
                self._test_labels =  data["test_labels"]
                self._test_data_size = data["test_data_size"]
                self._test_dataset = data["test_dataset"]
                del data

    def save(self, dataset_filename="CXR_png.pickle", overwrite=False):
        if self._dataset is None:
            print("Dataset is empty. Run load_images before saving.")
            return

        data = {"dataset": self._dataset,
                "labels": self._labels,
                "valid_images_count": self._valid_images_count,
                "width": self._image_width,
                "height": self._image_height,
                "convert_to_gray": self._convert_to_gray,
                "folder": self._folder,
                "test_dataset": self._test_dataset,
                "test_labels": self._test_labels,
                "test_data_size": self._test_data_size}

        if overwrite is True:
            if os.path.isfile(dataset_filename):
                os.remove(dataset_filename)
        try:
            with open(dataset_filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', dataset_filename, ':', e)

    def random_images(self, count, test_images=False):
        if test_images is True and self._test_data_size == 0:
            print("0 images in the test dataset. Use separate_test_dataset() to load images to test dataset.")
            return None

        if self._valid_images_count == 0:
            print("0 images in the dataset. Use load_images or load_from_pickle to load images to dataset.")
            return None
        num = 0
        dataset = np.ndarray(shape=(count,
                                    self._image_width,
                                    self._image_height,
                                    1 if self._convert_to_gray is True else 3), dtype=np.float32)
        labels = np.ndarray(shape=(count), dtype=np.int32)

        if test_images is True:
            for i in np.random.randint(low=0, high=self._test_data_size, size=count):
                dataset[num, ...] = self._test_dataset[i, ...]
                labels[num] = self._test_labels[i]
                num += 1
        else:
            for i in np.random.randint(low=0, high=self._valid_images_count, size=count):
                dataset[num, ...] = self._dataset[i, ...]
                labels[num] = self._labels[i]
        return dataset, labels, num

    def separate_test_dataset(self, num_of_test_images):
        if self._test_data_size != 0:
            print "Test files of size %s is already present in test dataset." % self._test_data_size
            return None
        if num_of_test_images >= self._valid_images_count:
            print("Dataset dont possess that many images.")
            return None
        if self._dataset is None:
            print("0 images in dataset. Add images via load_images.")
            return None
        self._test_data_size = num_of_test_images
        self._test_dataset = np.ndarray(shape=(num_of_test_images,
                                    self._image_width,
                                    self._image_height,
                                    1 if self._convert_to_gray is True else 3), dtype=np.float32)
        self._test_labels = np.ndarray(shape=(num_of_test_images), dtype=np.int32)

        self._test_dataset = self._dataset[-num_of_test_images:, ...]
        self._test_labels = self._labels[-num_of_test_images:]
        self._valid_images_count -= self._test_data_size
        self._dataset = self._dataset[0:self._valid_images_count, :, :, :]
        self._labels = self._labels[0:self._valid_images_count]

        print "self._test_data_size %s"% str(self._test_data_size)
        print "self._valid_count %s"% str(self._valid_images_count)
        print "self._labels.shape %s"% str(self._labels.shape)
        print "self._dataset.shape %s"% str(self._dataset.shape)
        print "self._test_labels.shape %s"% str(self._test_labels.shape)
        print "self._test_dataset.shape %s"% str(self._test_dataset.shape)

