import librosa
from os import listdir
from os.path import join
import random
import math
import numpy as np


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
noises.append([])


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
    if (len(samples) != sr):
        return results, filepath, class_id
    for item in noises:
        new_samples = generate_noisy_sample(samples, item)
        mfcc = get_mfcc(new_samples, sr)
        results.append(mfcc)
    return results, filepath, class_id
