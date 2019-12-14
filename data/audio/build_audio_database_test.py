import sys

sys.path.append("../lib")
import os
import librosa
import numpy as np
import utils
import itertools
import time
import random
import math
import scipy.io.wavfile as wavfile
import shutil
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--start_range', type=int)
parser.add_argument('--stop_range', type=int)
parser.add_argument('--num_speaker', type=int)
parser.add_argument('--max_num_sample', type=int)
parser.add_argument('--delete_old_data', type=str)
args = parser.parse_args()

# Parameter
SAMPLE_RANGE = (args.start_range, args.stop_range)  # data usage to generate database

WAV_REPO_PATH = "../dataset/norm_audio_test"
DATABASE_REPO_PATH = '../dataset/AV_model_database_test'
FRAME_LOG_PATH = '../dataset/valid_frame_test.txt'

NUM_SPEAKER = args.num_speaker
MAX_NUM_SAMPLE = args.max_num_sample


if args.delete_old_data == 'true':
    if os.path.isdir(DATABASE_REPO_PATH):
        shutil.rmtree(DATABASE_REPO_PATH, ignore_errors=True)

    os.mkdir(DATABASE_REPO_PATH)

elif args.delete_old_data == 'false':
    if not os.path.isdir(DATABASE_REPO_PATH):
        os.mkdir(DATABASE_REPO_PATH)


# time measure decorator
def timit(func):
    def cal_time(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        tac = time.time()
        print(func.__name__, 'running time: ', (tac - tic), 'ms')
        return result

    return cal_time


# create directory to store database
def init_dir(path=DATABASE_REPO_PATH):

    if not os.path.isdir('%s/mix' % path):
        os.mkdir('%s/mix' % path)

    if not os.path.isdir('%s/single' % path):
        os.mkdir('%s/single' % path)

    if not os.path.isdir('%s/mix_wav' % path):
        os.mkdir('%s/mix_wav' % path)


@timit
def generate_path_list(sample_range=SAMPLE_RANGE, repo_path=WAV_REPO_PATH, frame_path=FRAME_LOG_PATH):
    '''

    :param sample_range:
    :param repo_path:
    :return: 2D array with idx and path (idx_wav,path_wav)
    '''
    audio_path_list = []
    frame_set = set()

    with open(frame_path, 'r') as f:
        frames = f.readlines()

    for i in range(len(frames)):
        frame = frames[i].replace('\n', '').replace('frame_', '')
        frame_set.add(int(frame))

    for i in range(sample_range[0], sample_range[1]+1):
        # print('\rchecking...%d' % int(frame), end='')
        path = repo_path + '/trim_audio_test%d.wav' % i
        if os.path.exists(path) and (i in frame_set):
            audio_path_list.append((i, path))
    print('\nlength of the path list: ', len(audio_path_list))
    return audio_path_list


# data generate function
def single_audio_to_npy(audio_path_list, database_repo=DATABASE_REPO_PATH, fix_sr=16000):
    for idx, path in audio_path_list:
        print('\rsingle npy generating... %d' % ((idx / len(audio_path_list)) * 100), end='')
        data, _ = librosa.load(path, sr=fix_sr)
        data = utils.fast_stft(data, power=True)
        name = 'single-%06d' % idx
        with open('%s/single_TF.txt' % database_repo, 'a') as f:
            f.write('%s.npy' % name)
            f.write('\n')
        np.save(('%s/single/%s.npy' % (database_repo, name)), data)
    print()


# split single TF data to different part in order to mix
def split_to_mix(audio_path_list, database_repo=DATABASE_REPO_PATH, partition=2):
    # return split_list : (part1,part2,...)
    # each part : (idx,path)
    length = len(audio_path_list)
    if length == 0:
        return []
    part_len = length // partition
    if part_len == 1 and length%partition!=0:
        return []
    head = 0
    part_idx = 0
    split_list = []
    while (head + part_len) <= length:
        part = audio_path_list[head:(head + part_len)]
        split_list.append(part)
        with open('%s/single_TF_part%d.txt' % (database_repo, part_idx), 'a') as f:
            for idx, _ in part:
                name = 'single-%06d' % idx
                f.write('%s.npy' % name)
                f.write('\n')
        head += part_len
        part_idx += 1
    return split_list


# mix single TF data
def all_mix(split_list, database_repo=DATABASE_REPO_PATH, partition=2, max_samples = 1000):
    assert len(split_list) == partition
    print('mixing data...')
    num_mix = 1
    for part in split_list:
        num_mix *= len(part)
    print('number of mix data; ', num_mix)

    part_len = len(split_list[-1])
    idx_list = [x for x in range(part_len)]
    combo_idx_list = itertools.product(idx_list, repeat=partition)
    for num_mix_check,combo_idx in enumerate(combo_idx_list):
        single_mix(combo_idx, split_list, database_repo)
        if num_mix_check > max_samples:
            break
        print('\rnum of completed mixing audio : %d' % int(num_mix_check+1), end='')
    print()


# mix several wav file and store TF domain data with npy
def single_mix(combo_idx, split_list, database_repo):
    assert len(combo_idx) == len(split_list)
    mix_rate = 1.0 / float(len(split_list))
    wav_list = []
    prefix = "mix"
    mid_name = ""
    dataset_line = ""
    dataset_line_mid_name = ''

    for part_idx in range(len(split_list)):
        idx, path = split_list[part_idx][combo_idx[part_idx]]
        wav, _ = librosa.load(path, sr=16000)
        wav_list.append(wav)
        mid_name += '-%06d' % idx
        dataset_line_mid_name += 'single-%06d' % idx + '.npy '

    # mix wav file
    mix_wav = np.zeros_like(wav_list[0])
    for wav in wav_list:
        mix_wav += wav * mix_rate

    # save mix wav file
    wav_name = prefix + mid_name + '.wav'
    wavfile.write('%s/mix_wav/%s' % (database_repo, wav_name), 16000, mix_wav)

    # transfer mix wav to TF domain
    F_mix = utils.fast_stft(mix_wav, power=True)
    name = prefix + mid_name + ".npy"
    store_path = '%s/mix/%s' % (database_repo, name)

    # save mix as npy file
    np.save(store_path, F_mix)

    # save mix log
    with open('%s/mix_log.txt' % database_repo, 'a') as f:
        f.write(name)
        f.write("\n")

    dataset_line += name
    dataset_line += ' '
    dataset_line += dataset_line_mid_name
    with open('%s/dataset.txt' % database_repo, 'a') as f:
        f.write(dataset_line)
        f.write('\n')

def test_data_range(dataset_log_path, data_range=[0, 50000],database_repo=DATABASE_REPO_PATH):
    with open(dataset_log_path, 'r') as f:
        data_log = f.read().splitlines()

    if data_range[1] > len(data_log):
        data_range[1] = len(data_log)
    samples = data_log[data_range[0]:data_range[1]]

    test = samples[:]

    with open('%s/dataset_test.txt' % database_repo, 'a') as f:
        for line in test:
            f.write(line)
            f.write('\n')


if __name__ == "__main__":
    init_dir()
    audio_path_list = generate_path_list()
    single_audio_to_npy(audio_path_list)
    split_list = split_to_mix(audio_path_list, partition=NUM_SPEAKER)
    all_mix(split_list, partition=NUM_SPEAKER, max_samples = MAX_NUM_SAMPLE)

    dataset_log_path = '%s/dataset.txt' % DATABASE_REPO_PATH
    test_data_range(dataset_log_path, data_range=[0, MAX_NUM_SAMPLE])

    dataset_log_path_test = '%s/dataset_test.txt' % DATABASE_REPO_PATH
    dataset_log_path_test_AV = '%s/AVdataset_test.txt' % DATABASE_REPO_PATH

    with open(dataset_log_path_test, 'r') as t:
        lines = t.readlines()
        for line in lines:
            info = line.strip().split('.')
            num1 = info[0].strip().split('-')[1]
            num2 = info[0].strip().split('-')[2]

            newline = line.strip() + ' ' + num1 + '_face_emb.npy' + ' ' + num2 + '_face_emb.npy\n'
            with open(dataset_log_path_test_AV, 'a') as f:
                f.write(newline)