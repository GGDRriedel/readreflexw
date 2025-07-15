import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch, coherence

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class GPRNoiseAnalyzer:
    def __init__(self, traces, sampling_rate=1e9, time_window=None):
        """
        Advanced noise analysis for GPR data
        
        Parameters:
        -----------
        traces : numpy.ndarray
            GPR traces with shape (time_samples, num_traces)
        sampling_rate : float
            Sampling rate in Hz
        time_window : float
            Time window in seconds
        """
        self.traces = traces
        self.n_samples, self.n_traces = traces.shape
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        if time_window is None:
            self.time_window = self.n_samples * self.dt
        else:
            self.time_window = time_window
            
        self.time_axis = np.linspace(0, self.time_window, self.n_samples)
        self.freq_axis = fftfreq(self.n_samples, self.dt)
        self.freq_axis_positive = self.freq_axis[:self.n_samples//2]
        
    def detect_systematic_noise(self, threshold_percentile=95):
        """
        Detect systematic noise patterns that appear consistently across traces
        """
        # Calculate average trace
        avg_trace = np.mean(self.traces, axis=1)
        
        # Calculate correlation of each trace with average
        correlations = []
        for i in range(self.n_traces):
            corr = np.corrcoef(self.traces[:, i], avg_trace)[0, 1]
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # Find traces with unusually high correlation (systematic noise)
        threshold = np.percentile(correlations, threshold_percentile)
        systematic_noise_traces = np.where(correlations > threshold)[0]
        
        return {
            'avg_trace': avg_trace,
            'correlations': correlations,
            'threshold': threshold,
            'systematic_traces': systematic_noise_traces,
            'systematic_percentage': len(systematic_noise_traces) / self.n_traces * 100
        }
    
    def coherent_noise_analysis(self, window_size=1000):
        """
        Analyze coherent noise patterns across adjacent traces
        """
        coherence_values = []
        frequencies = None
        
        # Calculate coherence between adjacent traces
        for i in range(0, self.n_traces - 1, window_size):
            end_idx = min(i + window_size, self.n_traces - 1)
            
            if end_idx > i:
                f, coh = coherence(
                    self.traces[:, i], 
                    self.traces[:, end_idx], 
                    fs=self.sampling_rate,
                    nperseg=min(256, self.n_samples//4)
                )
                coherence_values.append(coh)
                if frequencies is None:
                    frequencies = f
        
        coherence_values = np.array(coherence_values)
        avg_coherence = np.mean(coherence_values, axis=0)
        
        # Find frequency bands with high coherence
        high_coherence_freqs = frequencies[avg_coherence > 0.7]
        
        return {
            'frequencies': frequencies,
            'coherence_matrix': coherence_values,
            'avg_coherence': avg_coherence,
            'high_coherence_freqs': high_coherence_freqs
        }
    
    def periodic_noise_detection(self, min_period=2, max_period=100):
        """
        Detect periodic noise patterns in the spatial domain
        """
        # Calculate autocorrelation for each time sample across traces
        autocorr_results = []
        
        for t_idx in range(0, self.n_samples, 10):  # Sample every 10th time point
            trace_slice = self.traces[t_idx, :]
            
            # Calculate autocorrelation
            autocorr = np.correlate(trace_slice, trace_slice, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            autocorr_results.append(autocorr[:max_period])
        
        autocorr_results = np.array(autocorr_results)
        avg_autocorr = np.mean(autocorr_results, axis=0)
        
        # Find peaks in autocorrelation (periodic patterns)
        peaks, _ = signal.find_peaks(avg_autocorr[min_period:], height=0.3)
        periodic_distances = peaks + min_period
        
        return {
            'autocorr_matrix': autocorr_results,
            'avg_autocorr': avg_autocorr,
            'periodic_distances': periodic_distances,
            'lags': np.arange(max_period)
        }
    
    def powerline_interference_analysis(self, powerline_freq=50, harmonics=10):
        """
        Analyze powerline interference (50/60 Hz and harmonics)
        """
        # Define powerline frequencies and harmonics
        powerline_freqs = [powerline_freq * (i + 1) for i in range(harmonics)]
        
        interference_levels = []
        
        for trace_idx in range(self.n_traces):
            trace = self.traces[:, trace_idx]
            fft_result = fft(trace)
            power_spectrum = np.abs(fft_result)**2
            
            trace_interference = []
            for freq in powerline_freqs:
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(self.freq_axis - freq))
                
                # Average power in a small window around the frequency
                window = 3
                start_idx = max(0, freq_idx - window)
                end_idx = min(len(power_spectrum), freq_idx + window + 1)
                
                avg_power = np.mean(power_spectrum[start_idx:end_idx])
                trace_interference.append(avg_power)
            
            interference_levels.append(trace_interference)
        
        interference_levels = np.array(interference_levels)
        
        return {
            'powerline_freqs': powerline_freqs,
            'interference_levels': interference_levels,
            'avg_interference': np.mean(interference_levels, axis=0),
            'std_interference': np.std(interference_levels, axis=0)
        }
    
    def impulse_noise_detection(self, threshold_factor=3):
        """
        Detect impulse noise (spikes) in the data
        """
        # Calculate derivative to enhance impulses
        diff_traces = np.diff(self.traces, axis=0)
        
        # Find standard deviation for each trace
        std_values = np.std(diff_traces, axis=0)
        
        # Detect impulses
        impulse_locations = []
        impulse_amplitudes = []
        
        for trace_idx in range(self.n_traces):
            trace_diff = diff_traces[:, trace_idx]
            threshold = threshold_factor * std_values[trace_idx]
            
            impulse_mask = np.abs(trace_diff) > threshold
            impulse_times = np.where(impulse_mask)[0]
            impulse_amps = trace_diff[impulse_mask]
            
            impulse_locations.append(impulse_times)
            impulse_amplitudes.append(impulse_amps)
        
        # Calculate impulse statistics
        total_impulses = sum(len(locs) for locs in impulse_locations)
        impulse_rate = total_impulses / (self.n_traces * self.n_samples)
        
        return {
            'impulse_locations': impulse_locations,
            'impulse_amplitudes': impulse_amplitudes,
            'impulse_rate': impulse_rate,
            'threshold_factor': threshold_factor,
            'diff_traces': diff_traces
        }
    
    def environmental_noise_clustering(self, n_clusters=5):
        """
        Cluster traces based on noise characteristics to identify environmental patterns
        """
        # Extract noise features for each trace
        features = []
        
        for trace_idx in range(self.n_traces):
            trace = self.traces[:, trace_idx]
            
            # Statistical features
            mean_val = np.mean(trace)
            std_val = np.std(trace)
            skewness = stats.skew(trace)
            kurtosis = stats.kurtosis(trace)
            
            # Spectral features
            fft_result = fft(trace)
            power_spectrum = np.abs(fft_result)**2
            
            # Spectral centroid
            freqs = np.abs(self.freq_axis)
            spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
            
            # Spectral rolloff (95% of energy)
            cumsum_power = np.cumsum(power_spectrum)
            rolloff_idx = np.where(cumsum_power >= 0.95 * cumsum_power[-1])[0][0]
            spectral_rolloff = freqs[rolloff_idx]
            
            # High frequency noise level
            high_freq_power = np.mean(power_spectrum[self.n_samples//2:])
            
            features.append([
                mean_val, std_val, skewness, kurtosis,
                spectral_centroid, spectral_rolloff, high_freq_power
            ])
        
        features = np.array(features)
        
        # Normalize features
        features_norm = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_norm)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_norm)
        
        return {
            'features': features,
            'features_normalized': features_norm,
            'cluster_labels': cluster_labels,
            'kmeans_model': kmeans,
            'pca_features': features_pca,
            'pca_model': pca,
            'feature_names': ['Mean', 'Std', 'Skewness', 'Kurtosis', 
                             'Spectral Centroid', 'Spectral Rolloff', 'HF Power']
        }
    
    def rfi_detection(self, bandwidth_threshold=10e6):
        """
        Detect Radio Frequency Interference (RFI) - narrowband interference
        """
        # Calculate average power spectral density
        freq_power = []
        
        for trace_idx in range(0, self.n_traces, 100):  # Sample every 100th trace
            trace = self.traces[:, trace_idx]
            f, psd = welch(trace, fs=self.sampling_rate, nperseg=256)
            freq_power.append(psd)
        
        freq_power = np.array(freq_power)
        avg_psd = np.mean(freq_power, axis=0)
        std_psd = np.std(freq_power, axis=0)
        
        # Detect narrowband interference
        # Look for peaks that are significantly above background
        threshold = avg_psd + 3 * std_psd
        
        # Find peaks
        peaks, properties = signal.find_peaks(avg_psd, height=threshold, distance=10)
        
        # Calculate bandwidth of each peak
        peak_bandwidths = []
        rfi_frequencies = []
        
        for peak_idx in peaks:
            # Find half-maximum points
            peak_power = avg_psd[peak_idx]
            half_max = peak_power / 2
            
            # Find bandwidth
            left_idx = peak_idx
            right_idx = peak_idx
            
            while left_idx > 0 and avg_psd[left_idx] > half_max:
                left_idx -= 1
            while right_idx < len(avg_psd) - 1 and avg_psd[right_idx] > half_max:
                right_idx += 1
            
            bandwidth = f[right_idx] - f[left_idx]
            
            if bandwidth < bandwidth_threshold:  # Narrowband interference
                peak_bandwidths.append(bandwidth)
                rfi_frequencies.append(f[peak_idx])
        
        return {
            'frequencies': f,
            'avg_psd': avg_psd,
            'std_psd': std_psd,
            'threshold': threshold,
            'rfi_frequencies': rfi_frequencies,
            'rfi_bandwidths': peak_bandwidths,
            'all_peaks': peaks
        }
    
    def plot_comprehensive_noise_analysis(self, trace_sample_size=1000):
        """
        Create comprehensive noise analysis plots
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Limit traces for performance
        trace_indices = np.linspace(0, self.n_traces-1, 
                                   min(trace_sample_size, self.n_traces), 
                                   dtype=int)
        
        # 1. Systematic noise analysis
        plt.subplot(3, 4, 1)
        systematic_result = self.detect_systematic_noise()
        plt.plot(self.time_axis * 1e9, systematic_result['avg_trace'])
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude')
        plt.title('Average Trace (Systematic Noise)')
        plt.grid(True)
        
        # 2. Correlation distribution
        plt.subplot(3, 4, 2)
        plt.hist(systematic_result['correlations'], bins=50, alpha=0.7)
        plt.axvline(systematic_result['threshold'], color='r', linestyle='--', 
                   label=f'Threshold ({systematic_result["systematic_percentage"]:.1f}%)')
        plt.xlabel('Correlation with Average')
        plt.ylabel('Count')
        plt.title('Trace Correlation Distribution')
        plt.legend()
        plt.grid(True)
        
        # 3. Coherent noise analysis
        plt.subplot(3, 4, 3)
        coherent_result = self.coherent_noise_analysis()
        plt.plot(coherent_result['frequencies'] / 1e6, coherent_result['avg_coherence'])
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Coherence')
        plt.title('Average Coherence Between Traces')
        plt.axhline(0.7, color='r', linestyle='--', alpha=0.5)
        plt.grid(True)
        
        # 4. Periodic noise detection
        plt.subplot(3, 4, 4)
        periodic_result = self.periodic_noise_detection()
        plt.plot(periodic_result['lags'], periodic_result['avg_autocorr'])
        plt.scatter(periodic_result['periodic_distances'], 
                   periodic_result['avg_autocorr'][periodic_result['periodic_distances']], 
                   color='red', s=50, zorder=5)
        plt.xlabel('Lag (traces)')
        plt.ylabel('Autocorrelation')
        plt.title('Periodic Patterns Detection')
        plt.grid(True)
        
        # 5. Powerline interference
        plt.subplot(3, 4, 5)
        powerline_result = self.powerline_interference_analysis()
        plt.plot(powerline_result['powerline_freqs'], powerline_result['avg_interference'], 'o-')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Average Power')
        plt.title('Powerline Interference')
        plt.yscale('log')
        plt.grid(True)
        
        # 6. Impulse noise statistics
        plt.subplot(3, 4, 6)
        impulse_result = self.impulse_noise_detection()
        impulse_counts = [len(locs) for locs in impulse_result['impulse_locations']]
        plt.plot(trace_indices, [impulse_counts[i] for i in trace_indices])
        plt.xlabel('Trace Number')
        plt.ylabel('Number of Impulses')
        plt.title(f'Impulse Noise (Rate: {impulse_result["impulse_rate"]:.2e})')
        plt.grid(True)
        
        # 7. Environmental noise clustering
        plt.subplot(3, 4, 7)
        cluster_result = self.environmental_noise_clustering()
        scatter = plt.scatter(cluster_result['pca_features'][:, 0], 
                             cluster_result['pca_features'][:, 1], 
                             c=cluster_result['cluster_labels'], 
                             cmap='tab10', alpha=0.6)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Environmental Noise Clusters')
        plt.colorbar(scatter)
        
        # 8. RFI detection
        plt.subplot(3, 4, 8)
        rfi_result = self.rfi_detection()
        plt.semilogy(rfi_result['frequencies'] / 1e6, rfi_result['avg_psd'])
        plt.semilogy(rfi_result['frequencies'] / 1e6, rfi_result['threshold'], 'r--', alpha=0.7)
        if rfi_result['rfi_frequencies']:
            plt.scatter([f/1e6 for f in rfi_result['rfi_frequencies']], 
                       [rfi_result['avg_psd'][np.argmin(np.abs(rfi_result['frequencies'] - f))] 
                        for f in rfi_result['rfi_frequencies']], 
                       color='red', s=100, marker='x', linewidths=3)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power Spectral Density')
        plt.title('RFI Detection')
        plt.grid(True)
        
        # 9. Noise variance across traces
        plt.subplot(3, 4, 9)
        noise_variance = np.var(self.traces, axis=0)
        plt.plot(noise_variance)
        plt.xlabel('Trace Number')
        plt.ylabel('Variance')
        plt.title('Noise Variance Across Traces')
        plt.grid(True)
        
        # 10. Frequency-domain noise map
        plt.subplot(3, 4, 10)
        freq_map = []
        for i in range(0, self.n_traces, max(1, self.n_traces//100)):
            fft_result = fft(self.traces[:, i])
            power_spectrum = np.abs(fft_result)**2
            freq_map.append(power_spectrum[:self.n_samples//2])
        
        freq_map = np.array(freq_map)
        plt.imshow(freq_map.T, aspect='auto', origin='lower',
                  extent=[0, len(freq_map), 0, self.sampling_rate/2/1e6])
        plt.xlabel('Trace Index (sampled)')
        plt.ylabel('Frequency (MHz)')
        plt.title('Frequency-Domain Noise Map')
        plt.colorbar()
        
        # 11. Statistical distribution of amplitudes
        plt.subplot(3, 4, 11)
        sample_data = self.traces[:, ::max(1, self.n_traces//10)].flatten()
        plt.hist(sample_data, bins=100, alpha=0.7, density=True)
        plt.xlabel('Amplitude')
        plt.ylabel('Density')
        plt.title('Amplitude Distribution')
        plt.yscale('log')
        plt.grid(True)
        
        # 12. Trace-to-trace difference analysis
        plt.subplot(3, 4, 12)
        if self.n_traces > 1:
            trace_diff = np.diff(self.traces, axis=1)
            diff_variance = np.var(trace_diff, axis=0)
            plt.plot(diff_variance)
            plt.xlabel('Trace Difference Index')
            plt.ylabel('Variance')
            plt.title('Trace-to-Trace Difference Variance')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_noise_report(self):
        """
        Generate a comprehensive noise analysis report
        """
        print("="*60)
        print("          GPR NOISE ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"Dataset size: {self.n_traces} traces � {self.n_samples} samples")
        print(f"Sampling rate: {self.sampling_rate/1e6:.1f} MHz")
        print(f"Total data points: {self.n_traces * self.n_samples:,}")
        
        # Systematic noise
        systematic_result = self.detect_systematic_noise()
        print(f"\n1. SYSTEMATIC NOISE:")
        print(f"   - Traces with systematic patterns: {systematic_result['systematic_percentage']:.1f}%")
        
        # Coherent noise
        coherent_result = self.coherent_noise_analysis()
        high_coh_count = len(coherent_result['high_coherence_freqs'])
        print(f"\n2. COHERENT NOISE:")
        print(f"   - Frequencies with high coherence: {high_coh_count}")
        if high_coh_count > 0:
            print(f"   - Frequency range: {coherent_result['high_coherence_freqs'][0]/1e6:.1f} - {coherent_result['high_coherence_freqs'][-1]/1e6:.1f} MHz")
        
        # Periodic noise
        periodic_result = self.periodic_noise_detection()
        print(f"\n3. PERIODIC NOISE:")
        print(f"   - Periodic patterns detected: {len(periodic_result['periodic_distances'])}")
        if len(periodic_result['periodic_distances']) > 0:
            print(f"   - Primary periods: {periodic_result['periodic_distances'][:3]} traces")
        
        # Powerline interference
        powerline_result = self.powerline_interference_analysis()
        max_interference_idx = np.argmax(powerline_result['avg_interference'])
        print(f"\n4. POWERLINE INTERFERENCE:")
        print(f"   - Strongest interference at: {powerline_result['powerline_freqs'][max_interference_idx]:.0f} Hz")
        print(f"   - Relative power level: {powerline_result['avg_interference'][max_interference_idx]:.2e}")
        
        # Impulse noise
        impulse_result = self.impulse_noise_detection()
        print(f"\n5. IMPULSE NOISE:")
        print(f"   - Impulse rate: {impulse_result['impulse_rate']:.2e} impulses/sample")
        print(f"   - Total impulses detected: {sum(len(locs) for locs in impulse_result['impulse_locations'])}")
        
        # RFI
        rfi_result = self.rfi_detection()
        print(f"\n6. RADIO FREQUENCY INTERFERENCE:")
        print(f"   - RFI frequencies detected: {len(rfi_result['rfi_frequencies'])}")
        if rfi_result['rfi_frequencies']:
            print(f"   - RFI frequencies: {[f/1e6 for f in rfi_result['rfi_frequencies'][:5]]} MHz")
        
        # Environmental clustering
        cluster_result = self.environmental_noise_clustering()
        cluster_sizes = np.bincount(cluster_result['cluster_labels'])
        print(f"\n7. ENVIRONMENTAL PATTERNS:")
        print(f"   - Noise clusters identified: {len(cluster_sizes)}")
        print(f"   - Largest cluster: {np.max(cluster_sizes)} traces ({np.max(cluster_sizes)/self.n_traces*100:.1f}%)")
        
        # Overall assessment
        overall_snr = np.mean(np.abs(self.traces)) / np.std(self.traces)
        print(f"\n8. OVERALL ASSESSMENT:")
        print(f"   - Estimated SNR: {overall_snr:.2f}")
        print(f"   - Data quality: {'Good' if overall_snr > 3 else 'Moderate' if overall_snr > 1 else 'Poor'}")
        
        print("="*60)


# Example usage function
def example_noise_analysis():
    """Example of how to use the noise analyzer"""
    # Create synthetic noisy GPR data
    np.random.seed(42)
    n_samples, n_traces = 512, 1000
    
    # Create base signal
    t = np.linspace(0, 50e-9, n_samples)
    base_signal = np.sin(2*np.pi*200e6*t) * np.exp(-t/10e-9)
    
    # Add various noise types
    traces = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        signal = base_signal.copy()
        
        # Add white noise
        signal += np.random.normal(0, 0.1, n_samples)
        
        # Add systematic noise (coupling)
        if i % 10 == 0:
            signal += 0.05 * np.sin(2*np.pi*50*t)  # 50 Hz powerline
        
        # Add periodic spatial noise
        if i % 20 == 0:
            signal += 0.02 * np.sin(2*np.pi*100e6*t)
        
        # Add impulse noise
        if np.random.random() < 0.05:
            impulse_idx = np.random.randint(0, n_samples)
            signal[impulse_idx] += np.random.normal(0, 0.5)
        
        traces[:, i] = signal
    
    # Analyze noise
    analyzer = GPRNoiseAnalyzer(traces, sampling_rate=1e9)
    analyzer.generate_noise_report()
    analyzer.plot_comprehensive_noise_analysis()
    
    return analyzer

# Uncomment to run example:
noise_analyzer = example_noise_analysis()