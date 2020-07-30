from os import listdir
from os.path import join, isfile, isdir, normpath
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from process_data import process_file, generate_noisy_sample
import numpy as np
from itertools import zip_longest
import librosa
from random import randint


def __load_audio_filenames_with_class__(root_folder):
    classes = [item for item in listdir(root_folder) if isdir(
        join(root_folder, item)) and not item.startswith('_')]
    filenames = []
    class_ids = []
    for i in range(len(classes)):
        c = classes[i]
        class_filenames = __load_audio_filenames__(join(root_folder, c))
        filenames.extend(class_filenames)
        class_ids.extend([i] * len(class_filenames))
    return filenames, class_ids, classes


def __load_audio_filenames__(root_folder):
    filenames = []
    for entry in listdir(root_folder):
        full_path = join(root_folder, entry)
        if (isfile(full_path)):
            if (entry.endswith('.wav')):
                filenames.append(full_path)
        else:
            filenames.extend(__load_audio_filenames__(full_path))
    return filenames


def __load_subset_filenames__(root_folder, filename):
    subset_list = []
    with open(join(root_folder, filename)) as f:
        for line in f:
            line = line.strip()
            if (len(line) == 0):
                continue
            subset_list.append(normpath(join(root_folder, line)))
    return set(subset_list)


def load_data_from_folder(root_folder):
    filenames, class_ids, classes = __load_audio_filenames_with_class__(
        root_folder)
    dataset_size = len(filenames)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_validation = []
    y_validation = []
    pool = Pool(cpu_count() - 1)
    for (results, filepath, class_id, random_roll) in tqdm(pool.imap_unordered(process_file, zip_longest(filenames, class_ids)), total=dataset_size):
        filepath = normpath(filepath)
        is_testing = 1 <= random_roll and random_roll <= 10
        is_validation = 11 <= random_roll and random_roll <= 20
        for item in results:
            if (is_testing):
                X_test.append(item)
                y_test.append(class_id)
            elif (is_validation):
                X_validation.append(item)
                y_validation.append(class_id)
            else:
                X_train.append(item)
                y_train.append(class_id)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation)
    return X_train, y_train, X_test, y_test, X_validation, y_validation, classes
