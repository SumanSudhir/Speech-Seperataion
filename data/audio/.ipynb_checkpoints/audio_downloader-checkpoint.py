import sys
sys.path.append("../lib")
import AVHandler as avh
import pandas as pd
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--start_range', type=int)
parser.add_argument('--stop_range', type=int)
parser.add_argument('--type', type=str)
parser.add_argument('--delete_old_data', type=str)
args = parser.parse_args()

def m_link(youtube_id):
    # return the youtube actual link
    link = 'https://www.youtube.com/watch?v=' + youtube_id
    return link


def m_audio(loc, name, csv, start_idx, end_idx):
    # make concatenated audio following by the catalog from AVSpeech
    # loc       | the location for file to store
    # name      | name for the wav mix file
    # csv       | the csv file with link and information to be extracted
    # start_idx | the starting index of the audio to download and concatenate
    # end_idx   | the ending index of the audio to download and concatenate

    for i in range(start_idx, end_idx+1):
        f_name = name + str(i)
        link = m_link(csv.loc[i, 0])
        start_time = csv.loc[i, 1]
        end_time = start_time + 3.0
        avh.download(loc, f_name, link)
        avh.cut(loc, f_name, start_time, end_time)


# give the location of avspeech_train.csv and avspeech_test.csv in read_csv('path')
# which is downloaded from https://looking-to-listen.github.io/avspeech/download.html

if args.type == 'train':
    train_data_csv = pd.read_csv('../dataset/avspeech_train.csv', header=None)

    if args.delete_old_data == 'true':
        if os.path.isdir('../dataset/audio_train'):
            shutil.rmtree('../dataset/audio_train', ignore_errors=True)

        os.mkdir('../dataset/audio_train')

    elif args.delete_old_data == 'false':
        if not os.path.isdir('../dataset/audio_train'):
            os.mkdir('../dataset/audio_train')

    m_audio('../dataset/audio_train/', 'audio_train', train_data_csv, args.start_range, args.stop_range)

elif args.type == 'test':
    test_data_csv = pd.read_csv('../dataset/avspeech_test.csv', header=None)

    if args.delete_old_data == 'true':
        if os.path.isdir('../dataset/audio_test'):
            shutil.rmtree('../dataset/audio_test', ignore_errors=True)

        os.mkdir('../dataset/audio_test')

    elif args.delete_old_data == 'false':
        if not os.path.isdir('../dataset/audio_test'):
            os.mkdir('../dataset/audio_test')

    m_audio('../dataset/audio_test/', 'audio_test', test_data_csv, args.start_range, args.stop_range)
