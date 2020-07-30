import sounddevice as sd
import numpy as np
from train import get_tc_resnet_8
from os import listdir
from os.path import isdir, join
from keras.models import load_model
from process_data import get_mfcc


sr = 16000
audio_length = 2
blocksize = 100
sd.default.samplerate = sr
sd.default.channels = 1
sd.default.blocksize = blocksize
sd.default.latency = 'low'

root_folder = 'dataset'

classes = [item for item in listdir(root_folder) if isdir(
    join(root_folder, item)) and not item.startswith('_')]
num_classes = len(classes)

model = get_tc_resnet_14((321, 40), num_classes)
model.load_weights('weights.h5')
model.summary()

recent_signal = []

try:
    while True:
        input("Press Enter to start recording:")
        stream = sd.InputStream()
        stream.start()
        print("Say the word:")
        while True:
            data, overflowed = stream.read(blocksize)
            data = data.flatten()
            recent_signal.extend(data.tolist())
            if (len(recent_signal) >= sr * audio_length):
                recent_signal = recent_signal[:sr * audio_length]
                break
        stream.close()
        print("Recording finished! Result is:")
        mfcc = get_mfcc(np.asarray(recent_signal), sr)
        y_pred = model.predict(np.array([mfcc]))[0]
        result_id = np.argmax(y_pred)
        result_prob = y_pred[result_id]
        print(classes[result_id] + " " + str(result_prob))
        recent_signal = []
except KeyboardInterrupt:
    print('Record finished!')
