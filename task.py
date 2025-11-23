import numpy as np
import matplotlib.pyplot as plt
import random as rand
import numpy.fft as fft

Fb = 1.023e6
Fs = 4*Fb
DURATION  = 0.020
AMPLITUDE = 1
IF =  Fs / 4
LENGTH_P = 4092
LENGTH_D = 1023
OC_RATE = 500.0
NAV_RATE = 125.0
CC_RATE = 250.0

t = np.arange(0.0,DURATION,1.0/Fs)
N = t.size


def lfsr_sequence(taps,start_seq, length):
    reg = start_seq.copy()
    seq = []

    for i in range(length):
        seq.append(reg[-1])
        bit = 0
        for t in taps:
            bit = bit ^reg[-t]
        reg =  [bit] + reg[:-1]
    return np.array(seq)

def upsampl(seq,fb,fs,duration):
    N = int(fs*duration)
    spc = int(fs/fb)
    seq = np.repeat(seq,spc)
    return seq[:N]

def make_mp(N):
    return np.tile([0,1],N//2+2)[:N]

def PVU(data_chips,pilot_chips):
    total_chips = len(data_chips) + len(pilot_chips)
    result = np.zeros(total_chips)


    data_positions = np.arange(0, len(data_chips) * 2, 2)
    result[data_positions] = data_chips[:len(data_positions)]

    pilot_positions = np.arange(1, len(pilot_chips) * 4, 4)
    result[pilot_positions[:len(pilot_chips)]] = pilot_chips

    return result[result != 0]

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

samples_nav = int(Fs/NAV_RATE)
nav_bits = np.tile([1,0],int(np.ceil(N/samples_nav/2)))
nav_seq =np.repeat(nav_bits,samples_nav)[:N]

samples_cc = int(Fs/CC_RATE)
CC_seq = np.repeat(nav_seq,2)[:N]

samples_oc = int(Fs/OC_RATE)
oc_seq = np.repeat([0,1],samples_oc)
oc_seq = np.tile(oc_seq,  N // len(oc_seq) + 1)[:N]

chips_mod = 1-2*(oc_seq^nav_seq)

mp =1-2*make_mp(N)

L1OCd=chips_mod^L1OCd_sequence
L1OCp=mp^L1OCp_sequence

