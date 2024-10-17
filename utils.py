import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd

# Ignore warnings
import warnings 
warnings.filterwarnings('ignore')

def loading(filename, Fs_target):
    # Read the audio file and sampling rate
    Fs_target = 16000
    Fs, data = wav.read(filename)
    data = data[:, 0] # stereo to mono

    # Transform signal from int16 (-32768 to 32767) to float32 (-1,1)
    if type(data[0]) == np.int16:
        data = np.divide(data,32768,dtype=np.float32)

    # Make sure the sampling rate is 16kHz
    if not (Fs == Fs_target):
        data = sig.resample_poly(data,Fs_target,Fs)
        Fs = Fs_target

    return data, Fs

def windowing(data, frame_length, hop_size, windowing_function):
    data = np.array(data)
    number_of_frames = 1 + int(np.floor((len(data)-frame_length)/hop_size))
    frame_matrix = np.zeros((frame_length,number_of_frames))

    if windowing_function == 'rect':
        window = np.ones((frame_length))
    elif windowing_function == 'hann':
        window = np.hanning(frame_length)
    elif windowing_function == 'cosine':
        window = np.sqrt(np.hanning(frame_length))
    elif windowing_function == 'hamming':
        window = np.hamming(frame_length)
    else:
        os.error('Windowing function not supported')
        
    
    for i in range(number_of_frames):
        frame = np.zeros(frame_length) # Initialize frame as zeroes (zero padding)
        start = i*hop_size
        stop = np.minimum(start+frame_length,len(data))
        frame[0:stop-start] = data[start:stop]
        frame_matrix[:,i] = np.multiply(window,frame)   
    return frame_matrix



def WGN(data_len, snr):
    #Add white gaussian noise to the signal with the defined snr.

    noise = np.random.normal(0,1,data_len)
    pow_ratio = np.power(10,(snr/20))
    noise_red = noise*pow_ratio

    return noise_red




def read_targets():
    #Target files must be in the same directory
    with open('ground_truth','r') as f:
        data = f.read()
    
    targets = np.array([int(i) for i in data.split()])
    targets = targets.reshape((1,len(targets)))
    return targets