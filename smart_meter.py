import json
import os
import numpy as np

from signal_analyzer import SignalAnalyzer
class SmartMeter:
    def __init__(
        self,
        fs,                        # Sampling frequency [Hz]
        f0,                        # Fundamental frequency [Hz]
        timestemp,                 # Timestemp of signal
        cycles_per_window,         # Number of fundamental cycles per FFT window
        step_duration,             # Time shift between consecutive windows [s]
        aggregation_period,        # Temporal aggregation interval Ta [s]
        aggregation_method="mean", # Aggregation operator over time: "mean", "rms", "max"
        max_order=None,            # Maximum harmonic order to evaluate
        min_amplitude=None,        # Absolute amplitude threshold [same unit as signal]
        min_relative=None,         # Relative threshold (ratio to fundamental)
        output_file_path=None,     # File path to save results
    ):
        self.fs = fs
        self.f0 = f0

        self.timestemp = timestemp

        # Window (Tw)
        self.window_duration = cycles_per_window / f0
        self.window_samples = int(self.window_duration * fs)

        # Step
        self.step_samples = int(step_duration * fs)

        # Aggregation
        self.aggregation_period = aggregation_period
        self.aggregation_method = aggregation_method

        # Harmonic parameters
        self.max_order = max_order
        self.min_amplitude = min_amplitude
        self.min_relative = min_relative

        self.analyzer = SignalAnalyzer(fs)

        self.results = []

        self.output_file_path = output_file_path

    def process(self, v_signal, i_signal): #(self, signal)
        N = len(v_signal)

        temp_results = []
        result_aggregated = []
        count = 0
        scale_aggregated = int(self.aggregation_period / self.window_duration)
        current_timestemp = self.timestemp
        # --- 1. Análise contínua ---
        for start in range(0, N - self.window_samples + 1, self.step_samples):
            end = start + self.window_samples

            v_window = v_signal[start:end]
            i_window = i_signal[start:end]
            #window = signal[start:end]

            power = self.analyzer.compute_power(v_window, i_window)

            thdi = self.analyzer.compute_thd(i_window, self.f0)
            thdv = self.analyzer.compute_thd(v_window, self.f0)


            hi = self.analyzer.get_harmonics(
                i_window,
                f0=self.f0,
                max_order=self.max_order,
                min_amplitude=self.min_amplitude,
                min_relative=self.min_relative
            )

            hv = self.analyzer.get_harmonics(
                v_window,
                f0=self.f0,
                max_order=self.max_order,
                min_amplitude=self.min_amplitude,
                min_relative=self.min_relative
            )

            frequency = self.analyzer.estimate_f0(v_window)

            temp_results.append({
                "timestamp": current_timestemp,
                "Vrms": power["Vrms"],
                "Irms": power["Irms"],
                "P": power["P"],
                "Q": power["Q"],
                "S": power["S"],
                "fp": power["FP"],
                "frequency": frequency,
                "THDv": thdv,
                "THDi": thdi,
                "harmonics_v": hv,
                "harmonics_i": hi
            })

            current_timestemp += self.step_samples / self.fs

            count += 1
            if count % scale_aggregated == 0:
                result_aggregated.append(self._aggregate_block(temp_results))
                temp_results = []


        if temp_results:
            result_aggregated.append(self._aggregate_block(temp_results))

        self.results = result_aggregated

        if self.output_file_path is not None:
            with open(self.output_file_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.results,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    default=float
                )

    def _aggregate_harmonics(self, block):

        aggregated = {}

        # Percorre cada janela do bloco
        for harmonics in block:

            # Percorre cada ordem harmônica
            for order, amp in harmonics.items():

                if order not in aggregated:
                    aggregated[order] = []

                aggregated[order].append(amp)

        result = {}


        # Agrega cada ordem
        for order, values in aggregated.items():

            values = np.array(values)

            if self.aggregation_method == "mean":
                agg = np.mean(values)

            elif self.aggregation_method == "rms":
                agg = np.sqrt(np.mean(values**2))

            elif self.aggregation_method == "max":
                agg = np.max(values)

            else:
                raise ValueError("Invalid aggregation method")

            result[order] = float(np.round(agg, 6))

        return result
    def _aggregate_block(self, block):
        keys = ["Vrms", "Irms", "P", "Q", "S", "fp", "frequency", "THDv", "THDi"]

        result = {
          "timestamp": block[-1]["timestamp"]
        }

        # Agrega grandezas
        for k in keys:
          values = np.array([item[k] for item in block])

          if self.aggregation_method == "mean":
              agg = np.mean(values)

          elif self.aggregation_method == "rms":
              agg = np.sqrt(np.mean(values**2))

          elif self.aggregation_method == "max":
              agg = np.max(values)

          else:
              raise ValueError("Invalid aggregation method")

          result[k] = float(np.round(agg, 6))

        # Harmônicos
        result["harmonics_v"] = self._aggregate_harmonics([item["harmonics_v"] for item in block])

        result["harmonics_i"] = self._aggregate_harmonics([item["harmonics_i"] for item in block])

        return result
    def get_results(self):
        return self.results

    def reset(self):
        self.results = []
