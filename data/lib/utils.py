import numpy as np
import librosa
import os 
import operator
import itertools
from numpy import inf

#### Load Data ####
# windowing fft/ifft function
def stft(data, fft_size=512, step_size=160,padding=True):
    # short time fourier transform
    if padding == True:
        # for 16K sample rate data, 48192-192 = 48000
        pad = np.zeros(192,)
        data = np.concatenate((data,pad),axis=0)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)),np.hanning(fft_size-112),np.zeros((56,))),axis=0)
    win_num = (len(data) - fft_size) // step_size
    out = np.ndarray((win_num, fft_size), dtype=data.dtype)
    for i in range(win_num):
        left = int(i * step_size)
        right = int(left + fft_size)
        out[i] = data[left: right] * window
    F = np.fft.rfft(out, axis=1)
    return F

def istft(F, fft_size=512, step_size=160,padding=True):
    # inverse short time fourier transform
    data = np.fft.irfft(F, axis=-1)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)),np.hanning(fft_size-112),np.zeros((56,))),axis=0)
    number_windows = F.shape[0]
    T = np.zeros((number_windows * step_size + fft_size))
    for i in range(number_windows):
        head = int(i * step_size)
        tail = int(head + fft_size)
        T[head:tail] = T[head:tail] + data[i, :] * window
    if padding == True:
        T = T[:48000]
    return T

# combine FFT bins to mel frequency bins
def mel2freq(mel_data,sr,fft_size,n_mel,fmax=8000):
    matrix= librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    return np.dot(mel_data,matrix)

def freq2mel(f_data,sr,fft_size,n_mel,fmax=8000):
    pre_matrix = librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    matrix = pre_matrix.T / np.sum(pre_matrix.T,axis=0)
    return np.dot(f_data,matrix)

# directly time to mel domain transformation
def time_to_mel(data,sr,fft_size,n_mel,step_size,fmax=8000):
    F = stft(data,fft_size,step_size)
    M = freq2mel(F,sr,fft_size,n_mel,fmax=8000)
    return M

def mel_to_time(M,sr,fft_size,n_mel,step_size,fmax=8000):
    F = mel2freq(M,sr,fft_size,n_mel)
    T = istft(F,fft_size,step_size)
    return T

def real_imag_expand(c_data,dim='new'):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1],2))
        D[:,:,0] = np.real(c_data)
        D[:,:,1] = np.imag(c_data)
        return D
    if dim =='same':
        D = np.zeros((c_data.shape[0],c_data.shape[1]*2))
        D[:,::2] = np.real(c_data)
        D[:,1::2] = np.imag(c_data)
        return D

def real_imag_shrink(F,dim='new'):
    # dim = 'new' or 'same'
    # shrink the complex data to combine real and imag number
    F_shrink = np.zeros((F.shape[0], F.shape[1]))
    if dim =='new':
        F_shrink = F[:,:,0] + F[:,:,1]*1j
    if dim =='same':
        F_shrink = F[:,::2] + F[:,1::2]*1j
    return F_shrink

def power_law(data,power=0.3):
    # assume input has negative value
    mask = np.zeros(data.shape)
    mask[data>=0] = 1
    mask[data<0] = -1
    data = np.power(np.abs(data),power)
    data = data*mask
    return data

def fast_stft(data,power=True,**kwargs):
    # directly transform the wav to the input
    # power law = A**0.3 , to prevent loud audio from overwhelming soft audio
    output_ = real_imag_expand(stft(data))
    if power:
        output_ = power_law(output_)
    return output_

def fast_istft(F,power=True,**kwargs):
    # directly transform the frequency domain data to time domain data
    # apply power law
    if power:
        F = power_law(F,(1.0/0.3))
    T = istft(real_imag_shrink(F))
    
    return T

def generate_cRM(Y,S):
    '''

    :param Y: mixed/noisy stft
    :param S: clean stft
    :return: structed cRM
    '''
    M = np.zeros(Y.shape)
    epsilon = 1e-8
    # real part
    M_real = np.multiply(Y[:,:,0],S[:,:,0])+np.multiply(Y[:,:,1],S[:,:,1])
    square_real = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_real = np.divide(M_real,square_real+epsilon)
    M[:,:,0] = M_real
    # imaginary part
    M_img = np.multiply(Y[:,:,0],S[:,:,1])-np.multiply(Y[:,:,1],S[:,:,0])
    square_img = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_img = np.divide(M_img,square_img+epsilon)
    M[:,:,1] = M_img
    return M

def cRM_sigmoid_compress(M,K=10,C=0.1):
    '''
    Recall that the irm takes on vlaues in the range[0,1],compress the cRM with sigmoid
    :param M: crm (298,257,2)
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''

    numerator = np.exp(C*M)
    denominator = 1+np.exp(C*M)
    crm = K * np.divide(numerator,denominator)

    return crm

def cRM_sigmoid_recover(O,K=10,C=0.1):
    '''

    :param O: predicted compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return M : uncompressed crm
    '''

    numerator = K-O
    denominator = O
    M = -np.multiply((1.0/C),np.log(np.divide(numerator,denominator)))

    return M


def fast_cRM(Fclean,Fmix,K=10,C=0.1):   #http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.2016.pdf
    '''

    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''
    M = generate_cRM(Fmix,Fclean)
    crm = cRM_sigmoid_compress(M,K,C)
    return crm

def fast_icRM(Y,crm,K=10,C=0.1):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    M = cRM_sigmoid_recover(crm,K,C)
    S = np.zeros(np.shape(M))
    S[:,:,0] = np.multiply(M[:,:,0],Y[:,:,0])-np.multiply(M[:,:,1],Y[:,:,1])
    S[:,:,1] = np.multiply(M[:,:,0],Y[:,:,1])+np.multiply(M[:,:,1],Y[:,:,0])
    return S

def generate_one_sample(audio_path_list,num_speaker,fix_sr=16000,verbose=0):
    '''
    generate one sample from audios in the list

    :param audio_path_list: list contains path of the wav audio file
    :param num_speaker: specify the task for speech separation
    :param fix_sr: fix sample rate
    :return X_sample , y_sample
    '''

    # check path is exist
    for path in audio_path_list:
        if not os.path.exists(path):
            if verbose == 1:
                print('%s is not exist!'% path)
            return

    # initiate variables
    data_list = []
    sr_list = []
    F_list = []  # STFT list for each sample
    X_sample = np.empty(shape=(298,257,2))
    y_sample = np.empty(shape=(298,257,2,num_speaker))

    # import data
    for i in range(num_speaker):
        data, sr = librosa.load(audio_path_list[i],sr=fix_sr)
        data_list.append(data)
        sr_list.append(sr)

    # create mix audio according to mix rate
    mix_rate = 1.0 / float(num_speaker)
    mix = np.zeros(shape=data_list[0].shape)
    for data in data_list:
        np.add(mix,data*mix_rate)

    # transfrom data via STFT and several preprocessing function
    for i in range(num_speaker):
        F = fast_stft(data_list[i])
        F_list.append(F)
    F_mix = fast_stft(mix)
    X_sample = F_mix

    # create cRM for each speaker and fill into y_sample
    for i in range(num_speaker):
        cRM = np.divide(F_list[i], F_mix, out=np.zeros_like(F_list[i]), where=F_mix!=0)
        y_sample[:,:,:,i] = cRM

    # return values
    if verbose == 1:
        print('shape of X: ',X_sample.shape)
        print('shape of y: ',y_sample.shape)
    return X_sample, y_sample

def generate_dataset(sample_range,repo_path,num_speaker=2,**kwargs):
    '''
    A function to generate dataset
    :param sample_range: range of the sample to create the dataset
    :param repo_path: audio repository
    :param num_speaker: number of speaker to separate
    :return: X_data, y_data
    '''
    audio_path_list = []
    X_data = []
    y_data = []
    num_data = 0
    for i in range(sample_range[0],sample_range[1]):
        path = repo_path + '/trim_audio_train%d.wav'%i
        if os.path.exists(path):
            audio_path_list.append(path)

    combinations = itertools.combinations(audio_path_list,num_speaker)
    for combo in combinations:
        num_data += 1
        X_sample,y_sample = generate_one_sample(combo,num_speaker)
        X_data.append(X_sample)
        y_data.append(y_sample)

    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    print('number of the data generated: ',num_data)
    return X_data, y_data

#### normalization function ####
def min_max_norm(x):
    # x should be numpy M*N matrix , normalize the N axis
    return (x-np.min(x,axis=0)) / (np.max(x,axis=0)-np.min(x,axis=0))
