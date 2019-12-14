from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import datetime
import shutil

sys.path.append("../lib")
import AVHandler as avh
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_range', type=int)
parser.add_argument('--stop_range', type=int)
parser.add_argument('--type', type=str)
parser.add_argument('--delete_old_data', type=str)
parser.add_argument('--delete_video', type=str)
args = parser.parse_args()

def download_video_frames(loc, csv, start_idx, end_idx, rm_video, type):
    # Download each video and convert to frames immediately, can choose to remove video file
    # loc        | the location for downloaded file
    # csv        | the csv file with link and information to be extracted
    # start_idx  | the starting index of the video to download
    # end_idx    | the ending index of the video to download
    # rm_video   | boolean value for delete video and only keep the frames

    for i in range(start_idx, end_idx+1):
        command = 'cd %s;' % loc
        f_name = str(i)
        link = avh.m_link(csv.loc[i, 0])
        start_time = csv.loc[i, 1]
        end_time = start_time + 3.0
        start_time = datetime.timedelta(seconds=start_time)
        end_time = datetime.timedelta(seconds=end_time)
        command += 'ffmpeg -i $(youtube-dl -f ”mp4“ --get-url ' + link + ') ' + '-c:v h264 -c:a copy -ss %s -to %s %s.mp4;' \
                   % (start_time, end_time, f_name)
        # command += 'ffmpeg -i %s.mp4 -r 25 %s.mp4;' % (f_name, 'clip_' + f_name)  # convert fps to 25
        # command += 'rm %s.mp4;' % f_name

        # converts to frames
        # command += 'ffmpeg -i %s.mp4 -y -f image2  -vframes 75 ../frames/%s-%%02d.jpg;' % (f_name, f_name)
        command += 'ffmpeg -i %s.mp4 -vf fps=25 ../frames_%s/%s-%%02d.jpg;' % (f_name, type, f_name)
        # command += 'ffmpeg -i %s.mp4 ../frames/%sfr_%%02d.jpg;' % ('clip_' + f_name, f_name)

        if rm_video:
            command += 'rm %s.mp4' % f_name
        os.system(command)

if args.delete_video == 'true':
    delete_y_n = True
elif args.delete_video == 'false':
    delete_y_n = False

if args.type == 'train':

    if args.delete_old_data == 'true':
        if os.path.isdir('../dataset/video_train'):
            shutil.rmtree('../dataset/video_train', ignore_errors=True)
        if os.path.isdir('../dataset/frames_train'):
            shutil.rmtree('../dataset/frames_train', ignore_errors=True)

        os.mkdir('../dataset/video_train')
        os.mkdir('../dataset/frames_train')

    elif args.delete_old_data == 'false':
        if not os.path.isdir('../dataset/video_train'):
            os.mkdir('../dataset/video_train')
        if not os.path.isdir('../dataset/frames_train'):
            os.mkdir('../dataset/frames_train')        


    train_data_csv = pd.read_csv('../dataset/avspeech_train.csv', header=None)
    download_video_frames(loc='../dataset/video_train', csv=train_data_csv, start_idx=args.start_range, end_idx=args.stop_range, rm_video=delete_y_n,
                          type='train')

elif args.type == 'test':

    if args.delete_old_data == 'true':
        if os.path.isdir('../dataset/video_test'):
            shutil.rmtree('../dataset/video_test', ignore_errors=True)
        if os.path.isdir('../dataset/frames_test'):
            shutil.rmtree('../dataset/frames_test', ignore_errors=True)
            
        os.mkdir('../dataset/video_test')
        os.mkdir('../dataset/frames_test')

    elif args.delete_old_data == 'false':
        if not os.path.isdir('../dataset/video_test'):
            os.mkdir('../dataset/video_test')
        if not os.path.isdir('../dataset/frames_test'):
            os.mkdir('../dataset/frames_test')        

    test_data_csv = pd.read_csv('../dataset/avspeech_test.csv', header=None)
    download_video_frames(loc='../dataset/video_test', csv=test_data_csv, start_idx=args.start_range, end_idx=args.stop_range, rm_video=delete_y_n,
                          type='test')
