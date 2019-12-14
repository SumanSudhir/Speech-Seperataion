
# for tensorflow
from mtcnn.mtcnn import MTCNN 
import cv2
import pandas as pd
import os
import glob
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--start_range', type=int)
parser.add_argument('--stop_range', type=int)
parser.add_argument('--type', type=str)
parser.add_argument('--delete_old_data', type=str)
args = parser.parse_args()

#https://github.com/davidsandberg/facenet


def bounding_box_check(faces, x, y):
    # check the center
    for face in faces:
        bounding_box = face['box']
        if bounding_box[1] < 0:
            bounding_box[1] = 0
        if bounding_box[0] < 0:
            bounding_box[0] = 0
        if bounding_box[0] - 50 > x or bounding_box[0] + bounding_box[2] + 50 < x:
            # print('change person from')
            # print(bounding_box)
            # print('to')
            continue
        if bounding_box[1] - 50 > y or bounding_box[1] + bounding_box[3] + 50 < y:
            # print('change person from')
            # print(bounding_box)
            # print('to')
            continue
        return bounding_box


def face_detect(file, detector, frame_path, csv_data):
    name = file.replace('.jpg', '').split('-')
    log = csv_data.iloc[int(name[0])]
    x = log[3]
    y = log[4]

    img = cv2.imread('%s/%s' % (frame_path, file))
    x = img.shape[1] * x
    y = img.shape[0] * y
    faces = detector.detect_faces(img)
    # check if detected faces
    if len(faces) == 0:
        # print('no face detect: ' + file)
        return  # no face
    bounding_box = bounding_box_check(faces, x, y)
    if bounding_box is None:
        # print('face is not related to given coord: ' + file)
        return
    # print(file, " ", bounding_box)
    # print(file, " ", x, y)
    crop_img = img[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
    crop_img = cv2.resize(crop_img, (160, 160))
    cv2.imwrite('%s/frame_' % output_dir + name[0] + '_' + name[1] + '.jpg', crop_img)
    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(crop_img)
    # plt.show()


if args.type == 'train':

    train_data_csv = pd.read_csv('../dataset/avspeech_train.csv')
    frame_path = '../dataset/frames_train'
    output_dir = '../dataset/face_input_train'
    valid_frame_path = '../dataset/valid_frame_train.txt'
    detect_range = (args.start_range, args.stop_range)
    print(detect_range)
    open_file = open(valid_frame_path, 'a')

    if args.delete_old_data == 'true':
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

        os.mkdir(output_dir)

    elif args.delete_old_data == 'false':
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    detector = MTCNN()         
    for i in range(detect_range[0], detect_range[1]+1):
        print('Image: ', i)
        for j in range(1, 76):
            file_name = "%d-%02d.jpg" % (i, j)
            if not os.path.exists('%s/%s' % (frame_path, file_name)):
                print('cannot find input: ' + '%s/%s' % (frame_path, file_name))
                continue
            face_detect(file_name, detector, frame_path, train_data_csv)

    for i in range(detect_range[0], detect_range[1]+1):
        valid = True
        for j in range(1, 76):
            if os.path.exists(output_dir + "/frame_%d_%02d.jpg" % (i, j)) is False:
                path = output_dir + "/frame_%d_*.jpg" % i
                for file in glob.glob(path):
                    os.remove(file)
                valid = False
                print('frame %s is not valid' % i)
                break

        if valid:
            with open(valid_frame_path, 'a') as f:
                frame_name = "frame_%d" % i
                f.write(frame_name + '\n')

elif args.type == 'test':

    test_data_csv = pd.read_csv('../dataset/avspeech_test.csv')
    frame_path = '../dataset/frames_test'
    output_dir = '../dataset/face_input_test'
    valid_frame_path = '../dataset/valid_frame_test.txt'
    detect_range = (args.start_range, args.stop_range)

    open_file = open(valid_frame_path, 'a')

    if args.delete_old_data == 'true':
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

        os.mkdir(output_dir)

    elif args.delete_old_data == 'false':
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    detector = MTCNN()          
    for i in range(detect_range[0], detect_range[1]+1):
        for j in range(1, 76):
            file_name = "%d-%02d.jpg" % (i, j)
            if not os.path.exists('%s/%s' % (frame_path, file_name)):
                print('cannot find input: ' + '%s/%s' % (frame_path, file_name))
                continue
            face_detect(file_name, detector, frame_path, test_data_csv)

    for i in range(detect_range[0], detect_range[1]+1):
        valid = True
        for j in range(1, 76):
            if os.path.exists(output_dir + "/frame_%d_%02d.jpg" % (i, j)) is False:
                path = output_dir + "/frame_%d_*.jpg" % i
                for file in glob.glob(path):
                    os.remove(file)
                valid = False
                print('frame %s is not valid' % i)
                break

        if valid:
            with open(valid_frame_path, 'a') as f:
                frame_name = "frame_%d" % i
                f.write(frame_name + '\n')
