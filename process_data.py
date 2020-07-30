import librosa
from os import listdir
from os.path import join
import random
import math
import numpy as np
from random import seed, randint


AUDIO_LENGTH = 2

seed(163)


def __load_background_noises__(root_folder):
    noises = []
    noise_folder = join(root_folder, '_background_noise_')
    for item in listdir(noise_folder):
        if (not item.endswith('.wav')):
            continue
        samples, sr = librosa.load(join(noise_folder, item), sr=None)
        noises.append(samples)
    return noises


noises = __load_background_noises__('dataset')


def generate_noisy_sample(samples, noise):
    samples_length = len(samples)
    noise_length = len(noise)
    if (noise_length < samples_length):
        return samples
    noise_start = random.randint(0, noise_length - samples_length - 1)
    noise_part = noise[noise_start:noise_start + samples_length]
    noise_coeff = random.uniform(0.0, 0.1)
    audio_offset = math.floor(
        random.uniform(-samples_length * 0.1, samples_length * 0.1))
    new_samples = np.zeros((samples_length))
    if (audio_offset >= 0):
        new_samples[audio_offset:] = samples[:samples_length - audio_offset]
    else:
        new_samples[:samples_length + audio_offset] = samples[-audio_offset:]
    new_samples = noise_part * noise_coeff + \
        (1.0 - noise_coeff) * new_samples
    return new_samples


def get_mfcc(samples, sr):
    return librosa.feature.mfcc(samples, sr=sr, n_mfcc=40, n_fft=400, hop_length=100).transpose()


def process_file(argv):
    (filepath, class_id) = argv
    results = []
    samples, sr = librosa.load(filepath, sr=None)
    samples_len = len(samples)
    if (samples_len > sr * AUDIO_LENGTH):
        samples = samples[- sr * AUDIO_LENGTH:]
    elif (samples_len < sr * AUDIO_LENGTH):
        temp = np.zeros((sr * AUDIO_LENGTH))
        temp[:samples_len] = samples
        samples = temp
    mfcc = get_mfcc(samples, sr)
    results.append(mfcc)
    random_roll = randint(1, 100)
    is_testing = 1 <= random_roll and random_roll <= 10
    is_validation = 11 <= random_roll and random_roll <= 20
    if (not is_testing and not is_validation):
        for item in noises:
            new_samples = generate_noisy_sample(samples, item)
            mfcc = get_mfcc(new_samples, sr)
            results.append(mfcc)
    return results, filepath, class_id, random_roll
