import pyedflib
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# === Settings ===
edf_files = {


    "chb03_02": r"C:\Users\ASUS\PycharmProjects\Langchainmodels\chb01_fast_download\chb01_13.edf",




}
channel = 2  # You can adjust this
output_folder = r"C:\Users\ASUS\PycharmProjects\Langchainmodels\chb01_fast_download\dataset\interictal"
os.makedirs(output_folder, exist_ok=True)

# === Process Each Preictal File ===
for file_key, edf_path in edf_files.items():
    print(f"📂 Processing {file_key}.edf ...")

    # Load .edf
    f = pyedflib.EdfReader(edf_path)
    signal = f.readSignal(channel)
    sampling_rate = int(f.getSampleFrequency(channel))
    f._close()

    # 10-second windows
    window_size = 10 * sampling_rate
    total_windows = len(signal) // window_size
    print(f"📊 Total windows found: {total_windows}")

    for i in range(total_windows):
        start = i * window_size
        end = start + window_size
        segment = signal[start:end]

        if len(segment) < window_size:
            continue

        # Normalize
        segment = (segment - np.mean(segment)) / np.std(segment)

        # Generate spectrogram
        plt.figure(figsize=(4, 4))
        plt.specgram(segment, Fs=sampling_rate, NFFT=256, noverlap=128)
        plt.axis('off')

        # Save with label 1 (preictal)
        filename = f"{file_key}_seg_{i}_label_1.png"
        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"✅ Done with {file_key}: {total_windows} spectrograms saved.\n")

print(f"🎉 All preictal spectrograms saved in: {output_folder}")
