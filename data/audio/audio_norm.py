import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--delete_old_data', type=str)
args = parser.parse_args()

if args.type == 'train':

    if args.delete_old_data == 'true':
        if os.path.isdir('../dataset/norm_audio_train'):
            shutil.rmtree('../dataset/norm_audio_train', ignore_errors=True)

        os.mkdir('../dataset/norm_audio_train')

    elif args.delete_old_data == 'false':
        if not os.path.isdir('../dataset/norm_audio_train'):
            os.mkdir('../dataset/norm_audio_train')

    for audio_path in os.listdir('../dataset/audio_train'):
        path = '../dataset/audio_train/%s' % audio_path
        norm_path = '../dataset/norm_audio_train/%s' % audio_path
        if os.path.exists(path):
            audio, _ = librosa.load(path, sr=16000)
            max = np.max(np.abs(audio))
            norm_audio = np.divide(audio, max)
            wavfile.write(norm_path, 16000, norm_audio)

elif args.type == 'test':

    if args.delete_old_data == 'true':
        if os.path.isdir('../dataset/norm_audio_test'):
            shutil.rmtree('../dataset/norm_audio_test', ignore_errors=True)

        os.mkdir('../dataset/norm_audio_test')

    elif args.delete_old_data == 'false':
        if not os.path.isdir('../dataset/norm_audio_test'):
            os.mkdir('../dataset/norm_audio_test')

    for audio_path in os.listdir('../dataset/audio_test'):
        path = '../dataset/audio_test/%s' % audio_path
        norm_path = '../dataset/norm_audio_test/%s' % audio_path
        if os.path.exists(path):
            audio, _ = librosa.load(path, sr=16000)
            max = np.max(np.abs(audio))
            norm_audio = np.divide(audio, max)
            wavfile.write(norm_path, 16000, norm_audio)