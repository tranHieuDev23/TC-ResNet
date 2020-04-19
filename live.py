import sounddevice as sd
import numpy as np
from train import get_tc_resnet_14
from os import listdir
from os.path import isdir, join
from keras.models import load_model
from process_data import get_mfcc


sr = 16000
blocksize = 100
sd.default.samplerate = sr
sd.default.channels = 1
sd.default.blocksize = blocksize
sd.default.latency = 'low'

root_folder = 'dataset'

classes = [item for item in listdir(root_folder) if isdir(
    join(root_folder, item)) and not item.startswith('_')]
num_classes = len(classes)

model = get_tc_resnet_14((161, 40), num_classes, 1.5)
model.load_weights('weights.h5')


recent_signal = []
recent_count = 0
currently_silent = True
votes = [0] * num_classes


def is_silent():
    global recent_signal
    non_silence = len([item for item in recent_signal if item >= 0.01])
    return non_silence < 100


def audio_callback(indata, frames):
    global recent_signal, recent_count, model, currently_silent, votes
    indata = indata.flatten()
    recent_signal.extend(indata.tolist())
    recent_count += frames
    if (recent_count > sr):
        recent_signal = recent_signal[-sr:]
        recent_count = sr
    if (recent_count == sr):
        previously_silent = currently_silent
        currently_silent = is_silent()
        if (previously_silent and currently_silent):
            return
        if (not currently_silent):
            mfcc = get_mfcc(np.asarray(recent_signal), sr)
            y_pred = model.predict(np.array([mfcc]))[0]
            result_id = np.argmax(y_pred)
            result_pred = y_pred[result_id]
            if (result_pred >= 0.8):
                votes[result_id] += 1
        else:
            final_id = 0
            for i in range(1, num_classes):
                if (votes[i] > votes[final_id]):
                    final_id = i
            print(classes[final_id])
            votes = [0] * num_classes


with sd.InputStream() as stream:
    try:
        while True:
            data, overflowed = stream.read(blocksize)
            audio_callback(data, blocksize)
    except KeyboardInterrupt:
        print('Record finished')
