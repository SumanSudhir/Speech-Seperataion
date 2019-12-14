import sys
import numpy as np
from facenet_pytorch import InceptionResnetV1
import os
import argparse
import shutil
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--delete_old_data', type=str)
args = parser.parse_args()

# MODEL_PATH = 'FaceNet_keras/facenet_keras.h5'

if args.type == 'train':
    VALID_FRAME_LOG_PATH = '../dataset/valid_frame_train.txt'
    FACE_INPUT_PATH = '../dataset/face_input_train/'
    save_path = '../dataset/face1022_emb_train'

elif args.type == 'test':
    VALID_FRAME_LOG_PATH = '../dataset/valid_frame_test.txt'
    FACE_INPUT_PATH = '../dataset/face_input_test/'
    save_path = '../dataset/face1022_emb_test'

if args.delete_old_data == 'true':
    if os.path.isdir(save_path):
        shutil.rmtree(save_path, ignore_errors=True)

    os.mkdir(save_path)

elif args.delete_old_data == 'false':
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

# model = load_model(MODEL_PATH)
# avgPool_layer_model = Model(inputs=model.input,outputs=model.get_layer('AvgPool').output)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# print(list(model.children())[:-4])
avgPool_layer_model = nn.Sequential(*list(model.children())[:-4])

for param in avgPool_layer_model.parameters():
    param.requires_grad = False
    
avgPool_layer_model = avgPool_layer_model.eval().to(device)

lines = []
with open(VALID_FRAME_LOG_PATH, 'r') as f:
    lines = f.readlines()

for line in lines:
    embtmp = np.zeros((75, 1, 1792))
    headname = line.strip()
    tailname = ''
    for i in range(1, 76):
        if i < 10:
            tailname = '_0{}.jpg'.format(i)
        else:
            tailname = '_' + str(i) + '.jpg'
        picname = headname + tailname
        I = mpimg.imread(FACE_INPUT_PATH + picname)
        I_np = np.array(I)
        I_np = torch.from_numpy(I_np[np.newaxis, :, :, :].transpose(0, 3, 1, 2)).to(device).float()
        
        embtmp[i - 1, :] = avgPool_layer_model(I_np).detach().cpu().numpy()[0].reshape(1,1792)

    people_index = int(line.strip().split('_')[1])
    npname = '{:06d}_face_emb.npy'.format(people_index)
    print(npname)

    np.save(save_path + '/' + npname, embtmp)
    with open(save_path + '/faceemb_dataset.txt', 'a') as d:
        d.write(npname + '\n')

