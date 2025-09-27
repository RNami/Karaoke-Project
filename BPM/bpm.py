import os, sys
import sys
import numpy as np
from scipy import signal
from  matplotlib import pyplot as plt
import librosa
import IPython.display as ipd
import pandas as pd
sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6

def plot_sonify_novelty_beats(fn_wav, fn_ann, title=''):
    ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann, label='onset', header=0)
    df = pd.read_csv(fn_ann, sep=';', keep_default_na=False, header=None)
    beats_sec = df.values
    Fs = 22050
    x, Fs = librosa.load(fn_wav, Fs)
    x_duration = len(x)/Fs
    nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=256, gamma=1, M=10, norm=1)
    figsize=(8,1.5)
    fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k', figsize=figsize,
                title=title)
    libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                        nontime_axis=True, time_min=0, time_max=x_duration)
    plt.show()
    x_beats = librosa.clicks(beats_sec, sr=Fs, click_freq=1000, length=len(x))
    ipd.display(ipd.Audio(x + x_beats, rate=Fs))

#print('Carlos Gardel: Por Una Cabeza')
#fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_PorUnaCabeza_quarter.csv')
#fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_PorUnaCabeza.wav')
#plot_sonify_novelty_beats(fn_wav, fn_ann)

title = 'Borodin: String Quartet No. 2, 3rd movement'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Borodin-sec39_RWC_quarter.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Borodin-sec39_RWC.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)

title = 'Chopin: Op.68, No. 3'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Chopin.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Chopin.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)

title = 'Fauré: Op.15'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Faure_Op015-01-sec0-12_SMD126.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Faure_Op015-01-sec0-12_SMD126.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)