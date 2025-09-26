import re
import numpy as np


def parse_nti_fft(file_path:str)->np.ndarray:
    '''
    parse the NTI FFT measurement to a np.array
    arg: path to the measurement file
    Return: measurement:np.ndarray: with shape (3,n)
    [bands,max,live], n = number of frequency bands

    '''
    with open(file_path, 'r') as f:
        lines = f.readlines()
    f_bands = []
    max_values = []
    live_values = []
    for band in re.finditer(r"(\d+\.\d+)+", lines[-6]):
        f_bands.append(float(band.group()))

    for max_value in re.finditer(r"(\d+\.\d+)+", lines[-5]):
        max_values.append(max_value.group())

    for live_value in re.finditer(r"(\d+\.\d+)+", lines[-4]):
        live_values.append(live_value.group())

    measurement = np.array((f_bands, max_values, live_values), dtype=float)
    return measurement

def parse_nti_RT60(file_path:str)->np.ndarray:
    '''
    parse the NTI RT60 measurement to a np.array
    arg: path to the measurement file
    Return: measurement:np.ndarray: with shape (2,n)
    [bands,rt60], n = number of frequency bands

    '''
    bands = []
    rt_60 = []
    with open(file_path, 'r') as f:
        file = f.read()
    for match in  re.finditer(r'# RT60 Average Results\n\n(.*\n)+#\s', file):
        average_results = match.group()

    for match in re.finditer(r'(\d+).*(\d\.\d+)', average_results):
        bands.append(float(match.group(1)))
        rt_60.append(match.group(2))
    measurement = np.array((bands,rt_60), dtype=float)
    return measurement


