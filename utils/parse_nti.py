import re
import numpy as np

file = 'NTi_Measurements_2509/2025-09-25_RT60_000_Report.txt'

def parse_nti_fft(file_path:str)->np.ndarray:
    '''
    
    Return: measurement:np.ndarray: with shape (3,n)
    [bands,max,live], n = number of frequency bands

    '''
    with open(file_path, 'r') as f:
        lines = f.readlines()
    f_bands = []
    max_values = []
    live_values = []
    for band in re.finditer(r"(\d+\.\d+)+", lines[-6]):
        f_bands.append(band.group())

    for max_value in re.finditer(r"(\d+\.\d+)+", lines[-5]):
        max_values.append(max_value.group())

    for live_value in re.finditer(r"(\d+\.\d+)+", lines[-4]):
        live_values.append(live_value.group())

    measurement = np.array((f_bands, max_values, live_values))
    return measurement

def parse_nti_RT60(file_path:str)->np.ndarray:
    '''

    Return: measurement:np.ndarray: with shape (3,n)
    [bands,max,live], n = number of frequency bands

    '''
    with open(file_path, 'r') as f:
        file = f.read()
    for match in  re.finditer(r'# RT60 Average Results\n\n(.*\n)+#\s', file):
        print(match.group())


parse_nti_RT60(file)


