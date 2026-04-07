import os
import numpy as np
from PIL import Image

def load_dataset(dataset_path, target_size=(64, 64), max_classes=None):
    class_names = sorted(os.listdir(dataset_path))

    if max_classes:
        class_names = class_names[:max_classes]

    data = []
    labels = []

    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):
            image_path = os.path.join(class_path, file_name)

            try:
                image = Image.open(image_path).convert("L").resize(target_size)
                image_array = np.array(image, dtype=np.float64) / 255.0

                data.append(image_array)

                # one-hot
                label = np.zeros(len(class_names))
                label[class_index] = 1
                labels.append(label)

            except:
                continue

    return np.array(data), np.array(labels), class_names