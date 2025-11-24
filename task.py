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
OC_RATE = 250.0
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
            bit = bit ^reg[t-1]
        reg =  [bit] + reg[:-1]
    return np.array(seq)

def upsampl(seq,fb,fs,duration):
    N = int(fs*duration)
    spc = int(fs/fb)
    seq = np.repeat(seq,spc)
    return seq[:N]

def make_mp(N,fs,f_mp=2.046e6):
    samples_per_mp = int(fs / f_mp)
    mp = np.tile([0, 1], N // (2 * samples_per_mp) + 1)
    return np.repeat(mp, samples_per_mp)[:N]

def PVU(data_chips,pilot_chips):
    total_chips = len(data_chips) + len(pilot_chips)
    result = np.zeros(total_chips)

    data_positions = np.arange(0, len(data_chips) * 2, 2)
    result[data_positions] = data_chips[:len(data_positions)]

    pilot_positions = np.arange(1, len(pilot_chips) * 4, 4)
    result[pilot_positions[:len(pilot_chips)]] = pilot_chips

    return result[result != 0]
CHEEPS = int(Fs*DURATION)
print(CHEEPS)
# ДК для L1OCp
reg_strat_p = [0,0,0,0,1,1,0,0,0,1,0,1]
reg_p = [0,0,1,1,1,0]  #14 в двоичной системе

sequence_start = lfsr_sequence([6,8,11,12],reg_strat_p, LENGTH_P)
sequence_p = lfsr_sequence([1,6],reg_p,LENGTH_P)
xor_sequences_p = sequence_start^sequence_p
repeat_p = int(np.ceil(CHEEPS / LENGTH_P))
chips_p = np.tile(xor_sequences_p, repeat_p)[:CHEEPS]

# ДК для L1OCd
reg_start_d = [0,0,1,1,0,0,1,0,0,0]
reg_d = [0,0,0,0,0,0,1,1,1,0]

sequence_d = lfsr_sequence([3,7,9,10],reg_start_d,LENGTH_D)
sequence_start = lfsr_sequence([7,10],reg_d, LENGTH_D)
xor_sequences_d = sequence_start^sequence_d
repeat_d = int(np.ceil(CHEEPS / LENGTH_D))
chips_d = np.tile(xor_sequences_d, repeat_d)[:CHEEPS]

seq_p = 1-2*upsampl(chips_p,Fb,Fs,DURATION)
seq_d = 1-2*upsampl(chips_d,Fb,Fs,DURATION)


samples_cc = int(Fs/CC_RATE)
cc_bits = np.tile([1,0], int(np.ceil(N/samples_cc/2)))  # пример
CC_seq = np.repeat(cc_bits, samples_cc)[:N]

samples_oc = int(Fs/OC_RATE)
oc_seq = np.repeat([0,1],samples_oc)
oc_seq = np.tile(oc_seq,  N // len(oc_seq) + 1)[:N]

mod_seq =oc_seq^CC_seq
chips_mod = 1-2*mod_seq

mp = make_mp(N,Fs)
mp = 1-2*mp


L1OCd=chips_mod*seq_d
L1OCp=mp*seq_p

