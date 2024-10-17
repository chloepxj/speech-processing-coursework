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


