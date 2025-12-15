import numpy as np
from scipy.signal import welch, savgol_filter, find_peaks, butter, filtfilt, detrend
from scipy.stats import zscore

class PDREstimator:
    """
    Implements the automated PDR estimation algorithm described by Zibrandtsen & Kjaer (2021).
    Paper: Fully automatic peak frequency estimation of the posterior dominant rhythm...
    """
    
    def __init__(self, sfreq, o1_index, o2_index, prominence=5):
        """
        sfreq: Sampling frequency (Hz)
        o1_index: Index of channel O1 in the data array
        o2_index: Index of channel O2 in the data array
        """
        self.sfreq = sfreq
        self.o1_idx = o1_index
        self.o2_idx = o2_index
        self.prominence=prominence

    def fit(self, data):
        """
        Main pipeline corresponding to Fig 1[cite: 122].
        data: numpy array (n_channels, n_samples)
        """
        # 1. Preprocessing [cite: 76]
        # Re-reference to Common Average [cite: 77]
        data_car = data - np.mean(data, axis=0)
        
        # Channel Rejection (RMS based) [cite: 78, 79]
        valid_indices, clean_data = self._channel_rejection(data_car)
        
        # Check if O1/O2 survived rejection [cite: 80]
        # Note: We need to map original O1/O2 indices to new clean indices
        if self.o1_idx not in valid_indices or self.o2_idx not in valid_indices:
            return {"pdr_o1": None, "pdr_o2": None, "error": "O1/O2 rejected"}
        
        if len(valid_indices) < 15:
             return {"pdr_o1": None, "pdr_o2": None, "error": "<15 channels"}

        # Update indices for the cleaned dataset
        new_o1 = np.where(valid_indices == self.o1_idx)[0][0]
        new_o2 = np.where(valid_indices == self.o2_idx)[0][0]

        # 2. Segmentation [cite: 82]
        # Filter 0.5 - 70 Hz [cite: 87]
        filtered_data = self._bandpass_filter(clean_data, 0.5, 70)
        
        # Identify valid masks
        mask = self._get_segmentation_mask(filtered_data, new_o1, new_o2)
        
        # Extract data for O1 and O2 using mask
        data_o1 = filtered_data[new_o1, mask]
        data_o2 = filtered_data[new_o2, mask]
        
        # 3. Spectral Estimation & Peak Detection [cite: 101, 135]
        res_o1 = self._estimate_peak(data_o1)
        res_o2 = self._estimate_peak(data_o2)
        
        return {
            "pdr_o1": res_o1,
            "pdr_o2": res_o2,
            "data_usage_percent": (np.sum(mask) / data.shape[1]) * 100
        }

    def _channel_rejection(self, data):
        """
        Rejects bad channels based on RMS thresholds[cite: 78, 79].
        """
        # Calculate RMS per channel
        rms_values = np.sqrt(np.mean(data**2, axis=1))
        
        # Z-score exclusion (outside +/- 2 SD) [cite: 78]
        z_scores = zscore(rms_values)
        mask_z = np.abs(z_scores) <= 2
        
        # Absolute threshold exclusion (RMS > 1000) [cite: 79]
        mask_abs = rms_values <= 1000
        
        valid_mask = mask_z & mask_abs
        valid_indices = np.where(valid_mask)[0]
        
        return valid_indices, data[valid_indices, :]

    def _bandpass_filter(self, data, low, high):
        """Standard FIR/IIR filter implementation."""
        # Paper specifies FIR, but IIR (Butterworth) is standard in Python SciPy if FIR taps aren't specified.
        # We use a 4th order Butterworth here for efficiency.
        b, a = butter(4, [low, high], btype='band', fs=self.sfreq)
        return filtfilt(b, a, data, axis=1)

    def _get_segmentation_mask(self, data, o1_idx, o2_idx):
        """
        Selects noise-free intervals [cite: 82-95].
        """
        # Global Mean Field Power (GMFP) [cite: 88]
        gmfp = np.std(data, axis=0)
        
        # 1s Sliding Window RMS Envelopes 
        window_size = int(1.0 * self.sfreq)
        
        def sliding_rms(x, w):
            # Efficient sliding window RMS using pandas or convolution
            # Using convolution for pure numpy
            return np.sqrt(np.convolve(x**2, np.ones(w)/w, mode='same'))

        env_gmfp = sliding_rms(gmfp, window_size)
        env_o1 = sliding_rms(data[o1_idx], window_size)
        env_o2 = sliding_rms(data[o2_idx], window_size)
        
        # High pass envelopes at 0.1Hz [cite: 88]
        # (Implementing simple detrend/filter on envelope as implied)
        b, a = butter(2, 0.1, btype='high', fs=self.sfreq)
        env_gmfp = filtfilt(b, a, env_gmfp)
        env_o1 = filtfilt(b, a, env_o1)
        env_o2 = filtfilt(b, a, env_o2)

        # Threshold Calculation 
        # "mean of the maximum values of each envelope"
        max_vals = [np.max(env_gmfp), np.max(env_o1), np.max(env_o2)]
        threshold = np.mean(max_vals)
        
        # Adaptive Logic [cite: 91, 92]
        if threshold < 60:
            threshold *= 1.10
        
        # Check if threshold is below the mean of any envelope (Paper says "either envelope", implying O1/O2)
        if threshold < np.mean(env_o1) or threshold < np.mean(env_o2):
            threshold *= 1.50
            
        # Identify Segments [cite: 94]
        # Valid if envelopes < threshold
        is_valid = (env_o1 < threshold) & (env_o2 < threshold)
        
        # Keep only continuous intervals > 10s [cite: 94]
        # This requires finding connected components of 'True'
        min_samples = 10 * self.sfreq
        labeled_mask = np.zeros_like(is_valid)
        
        # Simple run-length encoding equivalent
        is_valid_int = is_valid.astype(int)
        diffs = np.diff(np.concatenate(([0], is_valid_int, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        total_valid_samples = 0
        final_mask = np.zeros_like(is_valid, dtype=bool)
        
        for s, e in zip(starts, ends):
            if (e - s) >= min_samples:
                final_mask[s:e] = True
                total_valid_samples += (e - s)
        
        # Fallback Logic: If < 2 mins found, use default window 
        if total_valid_samples < (120 * self.sfreq):
            # Paper: 60s start to 60s end (or 10 mins). 
            # Simplified: Take middle 10 mins or whole file if shorter
            mid = data.shape[1] // 2
            half_win = int(300 * self.sfreq) # 5 mins
            start = max(0, mid - half_win)
            end = min(data.shape[1], mid + half_win)
            final_mask[:] = False
            final_mask[start:end] = True
            
        return final_mask

    def _estimate_peak(self, signal_segment):
        """
        Spectral estimation and peak finding [cite: 101-107].
        """
        if len(signal_segment) == 0:
            return None
            
        # Welch's Method 
        # 4s Hamming, 50% overlap, Pad to 4x window length
        nperseg = int(4 * self.sfreq)
        noverlap = int(2 * self.sfreq)
        nfft = nperseg * 4 
        
        freqs, psd = welch(signal_segment, fs=self.sfreq, window='hamming', 
                           nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        # Convert to dB [cite: 103]
        psd_db = 10 * np.log10(psd)
        
        # Trim to 2-16 Hz [cite: 103]
        freq_mask = (freqs >= 2) & (freqs <= 16)
        freqs = freqs[freq_mask]
        psd_db = psd_db[freq_mask]
        
        # Savitzky-Golay Smoothing [cite: 104, 105]
        # 3rd order, 25 frames
        # Ensure window_length is odd and <= size of array
        sg_window = 25 if len(psd_db) > 25 else (len(psd_db) // 2 * 2 + 1)
        if sg_window > 3:
            psd_smooth = savgol_filter(psd_db, window_length=sg_window, polyorder=3)
        else:
            psd_smooth = psd_db

        # Peak Detection [cite: 106]
        # Min dist 0.5 Hz. Convert Hz to array indices.
        # freq resolution = sfreq / nfft? No, it's (sfreq/2) / (nfft/2) = sfreq/nfft
        freq_res = freqs[1] - freqs[0]
        dist_samples = int(0.5 / freq_res)
        
        peaks, properties = find_peaks(psd_smooth, distance=dist_samples, prominence=self.prominence)
        
        if len(peaks) == 0:
            return None
            
        # Select the highest peak (Paper implies finding "the" PDR)
        # Usually the one with highest prominence or amplitude
        best_peak_idx = peaks[np.argmax(properties['prominences'])]
        pdr_freq = freqs[best_peak_idx]
        
        return pdr_freq

# --- Example Usage ---
if __name__ == '__main__':
    import mne
    raw = mne.io.read_raw_edf("file.edf", preload=True)
    data = raw.get_data() # (n_channels, n_samples)
    sfreq = raw.info['sfreq']
    o1_idx = raw.ch_names.index('O1')
    o2_idx = raw.ch_names.index('O2')

    estimator = PDREstimator(sfreq, o1_idx, o2_idx, prominence=5)
    results = estimator.fit(data)
    print(f"O1 PDR: {results['pdr_o1']} Hz")
    print(f"O2 PDR: {results['pdr_o2']} Hz")