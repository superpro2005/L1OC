import numpy as np
import matplotlib.pyplot as plt
import random as rand
import numpy.fft as fft

Fb = 1023e6
Fs = 4*Fb
DURATION  = 0.020
AMPLITUDE = 1
IF =  Fs / 4
LENGTH_P = 4092
LENGTH_D = 1023

t = np.arange(0.0,DURATION,1.0/Fs)
N = t.size


def lfsr_sequence(taps,start_seq, length):
    reg = start_seq.copy()
    seq = []

    for i in range(length):
        seq.append(reg[-1])
        bit = 0
        for t in taps:
            bit = bit ^reg[t]
        reg =  [bit] + reg[:-1]
    return np.array(seq)

def upsampl(seq,fb,fs,duration):
    N = int(fs*duration)
    spc = int(fs/fb)
    seq = np.repeat(seq,spc)
    return seq[:N]

# ДК для L1OCp
# из ИКД L1
reg1_strat = [0,0,0,0,1,1,0,0,0,1,0,1]
reg_p = [0,0,1,1,1,0]  #14 в двоичной системе

sequence_start = lfsr_sequence([6,8,11,12],reg1_strat, LENGTH_P)
sequence_p = lfsr_sequence([1,6],reg_p,LENGTH_P)
xor_sequences_p = sequence_start^sequence_p
L1OCp_sequence = 1-2*xor_sequences_p

# ДК для L1OCd
reg2_start = [0,0,1,1,0,0,1,0,0,0]
sequence_d = lfsr_sequence([1,6],reg2_start,LENGTH_D)
sequence_start = lfsr_sequence([6,8,11,12],reg1_strat, LENGTH_D)
xor_sequences_d = sequence_start^sequence_d
L1OCd_sequence = 1-2*xor_sequences_d

seq_p = upsampl(L1OCp_sequence,Fb,Fs,DURATION)
seq_d = upsampl(L1OCd_sequence,Fb,Fs,DURATION)


