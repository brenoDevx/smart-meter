
import numpy as np

from scipy.fft import rfft, rfftfreq

class SignalAnalyzer:
    def __init__(self, fs):
        # Usa a freq de amostragem para definir o eixo da freq
        self.fs = fs

    def compute_spectrum(self, signal):
        # Define n pontos do sinal
        N = len(signal)

        #janela de hanning
        window = np.hanning(N)
        signal = signal * window

        # Pegamos só os valores reais e positivos da fft
        fft_vals = rfft(signal)
        freqs = rfftfreq(N, 1 / self.fs)  # Definimos as frequencias

        # Multiplicamos por 2 para compensar energia da metade negativa
        amplitudes = (2 / np.sum(window)) * np.abs(fft_vals)
        #amplitudes = (2 / N) * np.abs(fft_vals)

        # Como a amplitude na freq 0 não tem parte negativa não a duplicamos
        amplitudes[0] = amplitudes[0] / 2

        #dividindo todos por raiz de 2 exceto  o componente cc(harmonico 0)
        amplitudes[1:] = amplitudes[1:] / np.sqrt(2)

        return freqs, amplitudes

    def get_harmonics(self, signal, f0, max_order=None, min_amplitude=None, min_relative=None):

        # Extrai freq e amplitude das freq fundamentais
        freqs, amplitudes = self.compute_spectrum(signal)

        # Limite max baseado em nyquist
        nyquist_freq = self.fs / 2

        if max_order is None:
            max_order = int(nyquist_freq / f0)

        # Encontra o índice mais proximo da freq f0
        idx_fund = np.argmin(np.abs(freqs - f0))
        fundamental_amp = amplitudes[idx_fund]

        # Dicionário para armazenar resultados
        harmonics = {}

        for k in range(1, max_order + 1):
            target_freq = k * f0  # Frequencia alvo

            # Idx pega o indice e capta a amplitude dessa freq
            idx = np.argmin(np.abs(freqs - target_freq))
            amp = amplitudes[idx]

            # Evita harmonicos irrelevantes por amplitude
            if min_amplitude is not None and amp < min_amplitude:
                continue

            # Só aceita o mínimo relativo (ex: 5%)
            if min_relative is not None:
                if fundamental_amp == 0:
                    continue
                if (amp / fundamental_amp) < min_relative:
                    continue

            # Chave = ordem harmônica; valor = amplitude [V]
            harmonics[str(k)] = float(np.round(amp, 6))

        return harmonics

    def get_spectrum(self, signal):
        # Sinal completo
        return self.compute_spectrum(signal)

    def compute_rms(self, signal):
        # Calcula rms qualquer
        return (np.sqrt(np.mean(signal**2)))

    def compute_power(self, voltage_signal, current_signal):
        # Calculo de rms das grandezas
        voltage_rms = self.compute_rms(voltage_signal)
        current_rms = self.compute_rms(current_signal)

        # Calculo de potencia
        active_power = np.mean(voltage_signal  * current_signal)                    # Potência ativa
        apparent_power = voltage_rms * current_rms                                  # Potência aparente
        reactive_power = np.sqrt(max(apparent_power**2 - active_power**2, 0))       # Potência reativa
        power_factor = active_power / apparent_power if apparent_power != 0 else 0  # Fator de potência

        return {
            "Vrms": float(voltage_rms),
            "Irms": float(current_rms),
            "P": float(active_power),
            "Q": float(reactive_power),
            "S": float(apparent_power),
            "FP": float(power_factor)
        }

    def compute_thd(self, signal, f0):
        # Calcula THD qualquer
        voltage_rms = self.compute_rms(signal)

        # Retorna primeiro harmonico para calcular rms da amplitude
        freqs, amplitudes = self.get_spectrum(signal)

        # Econtra índice da amplitude da fundamental
        idx_fund = np.argmin(np.abs(freqs - f0))
        A1 = amplitudes[idx_fund]

        V1_rms = A1
        if V1_rms == 0:
          return 0

        thd = np.sqrt(max(voltage_rms**2 - V1_rms**2, 0)) / V1_rms

        return thd

    def estimate_f0(self, signal, f_min=10, f_max=100):
        """
        Estima a frequência fundamental encontrando o pico de magnitude
        dentro de uma faixa específica (ex: 10Hz a 100Hz para redes elétricas).
        """
        freqs, amplitudes = self.compute_spectrum(signal)

        # Filtra os índices que estão dentro da faixa de busca (f_min a f_max)
        mask = (freqs >= f_min) & (freqs <= f_max)

        if not np.any(mask):
            return 0.0

        search_freqs = freqs[mask]
        search_amps = amplitudes[mask]

        # Encontra o índice do valor máximo de amplitude nessa faixa
        idx_max = np.argmax(search_amps)
        return float(search_freqs[idx_max])

    def get_signal_info(self, signal, f0):
        # Pega todos os parametros do sinal

        N = len(signal)  # Numero de amostras
        Fs = self.fs     # Frequencia de amostragem
        Ts = 1 / Fs      # Período de amostragem

        # Espaçamento entre as bins
        df = Fs / N

        # Numero de ciclos da freq fundamental presente na janela amostrada
        Nc = (f0 * N) / Fs

        # Quantos harmonicos cabem antes de atingir a taxa de nyquist
        Hmax = int((Fs / 2) // f0)

        # Quanto sobra até nyquist depois do último harmônico
        Nm = (Fs / 2) - (Hmax * f0)

        return {
          "Fs": {
              "value": Fs,
              "unit": "Hz",
              "description": "Sampling frequency"
          },
          "Ts": {
              "value": Ts,
              "unit": "s",
              "description": "Sampling period (time between samples)"
          },
          "N": {
              "value": N,
              "unit": "samples",
              "description": "Total number of samples"
          },
          "f0": {
              "value": f0,
              "unit": "Hz",
              "description": "Fundamental frequency of the signal"
          },
          "df": {
              "value": df,
              "unit": "Hz",
              "description": "Spectral resolution (frequency spacing between FFT bins)"
          },
          "Nc": {
              "value": Nc,
              "unit": "cycles",
              "description": "Number of fundamental cycles within the sampled window"
          },
          "Hmax": {
              "value": Hmax,
              "unit": "order",
              "description": "Maximum harmonic order observable before Nyquist limit"
          },
          "Nm": {
              "value": Nm,
              "unit": "Hz",
              "description": "Nyquist margin (distance between highest harmonic and Nyquist frequency)"
          }
      }

