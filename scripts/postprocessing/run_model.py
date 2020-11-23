import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)

import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def predict(model_path, images_path, labels_path, output_path=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    category_index = label_map_util.create_category_index_from_labelmap(
        labels_path,
        use_display_name=True)
    model_fn = load_model(model_path)
    image_paths = get_image_paths(images_path)
    infer(model_fn, image_paths, category_index, output_path)


def load_model(path):
    print('Loading model')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Model loaded after {elapsed_time} seconds')
    return detect_fn


def get_image_paths(path):
    image_paths = []
    for _, _, filenames in os.walk(path):
        for f in filenames:
            image_path = os.path.abspath(os.path.join(path, f))
            if not image_path.endswith(('.JPG', '.PNG')):  # TODO: improve
                continue
            image_paths.append(image_path)
    return image_paths


def infer(model_fn, image_paths, category_index, save_path=None):
    for image_path in image_paths:
        print(f'Running inference for {os.path.basename(image_path)}')

        # Convert image to NumPy array
        image_np = np.array(Image.open(image_path))

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections[
            'detection_classes'].astype(
            np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        plt.imshow(image_np_with_detections)
        print('Done')
        if save_path:
            save_path_abs = os.path.join(save_path, os.path.basename(image_path))
            plt.savefig(save_path_abs)


predict(model_path='../../workspace/training_demo/exported_models/my_model/saved_model',
        images_path='../../workspace/training_demo/images/train',
        labels_path='../../workspace/training_demo/annotations/label_map.pbtxt',
        output_path=os.getcwd())