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


def PVU(data_chips, pilot_chips):
    # Обрезаем до минимальной длины и чередуем
    min_len = min(len(data_chips), len(pilot_chips))
    result = np.empty(min_len * 2)
    result[0::2] = data_chips[:min_len]   # четные позиции
    result[1::2] = pilot_chips[:min_len]  # нечетные позиции
    return result
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
reg_d =       [0,0,0,0,0,0,1,1,1,0]

sequence_d = lfsr_sequence([7,10],reg_start_d,LENGTH_D)
sequence_start = lfsr_sequence([3,7,9,10],reg_d, LENGTH_D)
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

PVU_L1OC = PVU(L1OCd,L1OCp)
carrier=np.cos(2*np.pi*IF*t[:len(PVU_L1OC)])
min_len = min(len(PVU_L1OC), len(carrier))
if_signal = AMPLITUDE*PVU_L1OC[:min_len]*carrier[:min_len]


def pltSPEC(signal, fs):
    N_fft = 2 ** int(np.ceil(np.log2(len(signal))))
    spectrum = np.fft.fft(signal, n=N_fft)
    spectrum_shifted = np.fft.fftshift(spectrum)
    freq = np.fft.fftfreq(N_fft, 1 / fs)
    freq_shifted = np.fft.fftshift(freq)

    spectrum_db = 20.0 * np.log10(np.abs(spectrum_shifted) + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.plot(freq_shifted / 1e6, spectrum_db)
    plt.title('Energy Spectrum (dB)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.xlim(-2.5, 2.5)
    plt.tight_layout()

def pltACF(signal, fs):
    corr_full = np.correlate(signal, signal, mode='full')
    corr = corr_full[len(signal)-1:]
    lags = np.arange(len(corr)) / fs * 1000.0

    corr = corr / (np.max(np.abs(corr)) + 1e-12)

    plt.figure(figsize=(6, 4))
    plt.plot(lags, corr)
    plt.title('Autocorrelation Function (normalized)')
    plt.xlabel('Lag (ms)')
    plt.ylabel('Normalized Correlation')
    plt.grid(True)
    plt.xlim(0, 10)
    plt.tight_layout()


def print_hex_edges(bits, name="seq"):
    b = np.array(bits).astype(int)
    f = b[:32]
    l = b[-32:]

    f_hex = hex(int("".join(map(str, f)), 2))[2:].upper().zfill(8)
    l_hex = hex(int("".join(map(str, l)), 2))[2:].upper().zfill(8)

    print(f"{name}: first={f_hex}, last={l_hex}")


print_hex_edges(chips_d, "L1OCd")
print_hex_edges(chips_p, "L1OCp")
pltSPEC(if_signal, Fs)
pltACF(if_signal, Fs)
plt.show()