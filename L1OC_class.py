import numpy as np
import matplotlib.pyplot as plt

class L1OC_sim:
    def __init__(self, Fb=1.023e6, Fs=None, duration=0.020, amplitude=1, IF=None,
                 LENGTH_P=4092, LENGTH_D=1023, OC_RATE=250.0, CC_RATE=250.0):
        self.Fb = Fb
        self.Fs =4*Fb
        self.DURATION = duration
        self.AMPLITUDE = amplitude
        self.IF = Fb
        self.LENGTH_P = LENGTH_P
        self.LENGTH_D = LENGTH_D
        self.OC_RATE = OC_RATE
        self.CC_RATE = CC_RATE
        self.t = np.arange(0.0, self.DURATION, 1.0/self.Fs)
        self.N = self.t.size
        self.CHEEPS = int(self.Fs*self.DURATION)

    def lfsr_sequence(self, taps, start_seq, length):
        reg = start_seq.copy()
        seq = []
        for _ in range(length):
            seq.append(reg[-1])
            bit = 0
            for t in taps:
                bit ^= reg[t-1]
            reg = [bit] + reg[:-1]
        return np.array(seq)

    def upsampl(self, seq, fb):
        N = int(self.Fs*self.DURATION)
        spc = int(self.Fs/fb)
        seq = np.repeat(seq, spc)
        return seq[:N]

    def make_mp(self, N, f_mp=2.046e6):
        samples_per_mp = int(self.Fs / f_mp)
        mp = np.tile([0,1], N // (2*samples_per_mp) + 1)
        return np.repeat(mp, samples_per_mp)[:N]

    def PVU(self, data_chips, pilot_chips):
        min_len = min(len(data_chips), len(pilot_chips))
        data_trim = data_chips[:min_len]
        pilot_trim = pilot_chips[:min_len]
        result = np.empty(min_len*2)
        result[0::2] = data_trim
        result[1::2] = pilot_trim
        return result

    def generate_sequences(self):
        reg_strat_p = [0,0,0,0,1,1,0,0,0,1,0,1]
        reg_p = [0,0,1,1,1,0] # HKA 14 остальное дополняем 0
        seq_start_p = self.lfsr_sequence([6,8,11,12], reg_strat_p, self.LENGTH_P)
        seq_p = self.lfsr_sequence([1,6], reg_p, self.LENGTH_P)
        xor_seq_p = seq_start_p ^ seq_p
        repeat_p = int(np.ceil(self.CHEEPS / self.LENGTH_P))
        chips_p = np.tile(xor_seq_p, repeat_p)[:self.CHEEPS]

        reg_start_d = [0,0,1,1,0,0,1,0,0,0]
        reg_d = [0,0,0,0,0,0,1,1,1,0] # HKA 14 остальное дополняем 0
        seq_start_d = self.lfsr_sequence([7,10], reg_start_d, self.LENGTH_D)
        seq_d = self.lfsr_sequence([3,7,9,10], reg_d, self.LENGTH_D)
        xor_seq_d = seq_start_d ^ seq_d
        repeat_d = int(np.ceil(self.CHEEPS / self.LENGTH_D))
        chips_d = np.tile(xor_seq_d, repeat_d)[:self.CHEEPS]

        seq_p_up = 1 - 2*self.upsampl(chips_p, self.Fb)
        seq_d_up = 1 - 2*self.upsampl(chips_d, self.Fb)

        samples_cc = int(self.Fs/self.CC_RATE)
        cc_bits = np.tile([1,0], int(np.ceil(self.N/samples_cc/2)))
        CC_seq = np.repeat(cc_bits, samples_cc)[:self.N]

        samples_oc = int(self.Fs/self.OC_RATE)
        oc_seq = np.repeat([0,1], samples_oc)
        oc_seq = np.tile(oc_seq, self.N // len(oc_seq) + 1)[:self.N]

        mod_seq = oc_seq ^ CC_seq
        chips_mod = 1 - 2*mod_seq
        mp = self.make_mp(self.N)
        mp = 1 - 2*mp

        self.L1OCd = chips_mod * seq_d_up
        self.L1OCp = mp * seq_p_up
        self.PVU_L1OC = self.PVU(self.L1OCd, self.L1OCp)
        self.carrier = np.cos(2*np.pi*self.IF*np.arange(len(self.PVU_L1OC))/self.Fs)
        self.if_signal = self.AMPLITUDE * self.PVU_L1OC * self.carrier
        self.chips_d = chips_d
        self.chips_p = chips_p

    def pltSPEC(self,signal):
        N_fft = 2 ** int(np.ceil(np.log2(len(signal))))
        spectrum = np.fft.fft(signal, n=N_fft)
        spectrum_shifted = np.fft.fftshift(spectrum)
        freq = np.fft.fftfreq(N_fft, 1 / self.Fs)
        freq_shifted = np.fft.fftshift(freq)

        spectrum_db = 20.0 * np.log10(np.abs(spectrum_shifted) + 1e-12)

        plt.figure(figsize=(8, 4))
        plt.plot(freq_shifted / 1e6, spectrum_db)
        plt.title('Спектр сигнала')
        plt.xlabel('Частота (MHz)')
        plt.ylabel('Амплитуда (dB)')
        plt.grid(True)
        plt.xlim(-2.5, 2.5)
        plt.ylim(0,100)
        plt.tight_layout()

    def pltACF(self,signal):
        corr_full = np.correlate(signal, signal, mode='full')
        corr = corr_full[len(signal) - 1:]
        time = np.arange(len(corr)) / self.Fs * 1000.0

        corr = corr / (np.max(corr))

        plt.figure(figsize=(6, 4))
        plt.plot(time, corr)
        plt.title('АКФ')
        plt.xlabel('время (ms)')
        plt.ylabel('Значение корреляции')
        plt.grid(True)
        plt.xlim(-.1,10)
        plt.ylim(0,1.2)
        plt.tight_layout()

    def print_hex_edges(self, bits, name="seq"):
        b = np.array(bits).astype(int)
        f = b[:32]
        l = b[-32:]
        f_hex = hex(int("".join(map(str, f)), 2))[2:].upper().zfill(8)
        l_hex = hex(int("".join(map(str, l)), 2))[2:].upper().zfill(8)
        print(f"{name}: Первые 32 символа: {f_hex}, Последние 32 символа: {l_hex}")

L1OC = L1OC_sim()
L1OC.generate_sequences()
L1OC.print_hex_edges(L1OC.chips_d, "L1OCd")
L1OC.print_hex_edges(L1OC.chips_p, "L1OCp")
L1OC.pltSPEC(L1OC.if_signal)
L1OC.pltACF(L1OC.if_signal)
plt.show()
