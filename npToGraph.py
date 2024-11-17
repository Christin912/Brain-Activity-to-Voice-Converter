# Save as npToGraph.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
import sys

def plot_eeg_data(npz_file):
    """
    Create comprehensive EEG visualization from NPZ data
    """
    print(f"Loading data from: {npz_file}")
    
    try:
        # Load the data
        data = np.load(npz_file)
        timestamps = data['timestamps']
        values = data['values']      # Raw ADC values
        
        print("\nFirst few ADC values:", values[:10])
        print(f"ADC value range: [{np.min(values)}, {np.max(values)}]")
        
        # Convert timestamps to seconds from start
        timestamps = (timestamps - timestamps[0]) / 1000.0
        
        # Calculate sampling rate
        sample_rate = len(timestamps) / timestamps[-1]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        # Adjust height ratios to make room for stats text
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 0.5], width_ratios=[3, 1])
        
        # Plot 1: EEG Signal (ADC Values) - spans both columns
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(timestamps, values, 'b-', linewidth=0.5, alpha=0.8)
        ax1.set_title('EEG Signal', fontsize=12, pad=10)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('ADC Value (0-1023)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spectrogram - spans both columns
        ax2 = plt.subplot(gs[1, :])
        try:
            frequencies, times, Sxx = signal.spectrogram(values, fs=sample_rate,
                                                       nperseg=min(256, len(values)//4),
                                                       noverlap=min(128, len(values)//8))
            
            # Add small constant to avoid log(0)
            Sxx = Sxx + 1e-10
            
            im = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx), 
                              shading='gouraud', cmap='viridis')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_title('Spectrogram', fontsize=12, pad=10)
            plt.colorbar(im, ax=ax2, label='Power/Frequency (dB/Hz)')
        except Exception as e:
            print(f"Warning: Could not generate spectrogram: {e}")
            ax2.text(0.5, 0.5, 'Could not generate spectrogram',
                    ha='center', va='center')
        
        # Plot 3: Signal Statistics - left column only
        ax3 = plt.subplot(gs[2, 0])
        sns.boxplot(x=values, ax=ax3, color='lightblue')
        ax3.set_title('ADC Value Distribution', fontsize=12, pad=10)
        ax3.set_xlabel('ADC Value')
        
        # Stats text in the bottom right cell of the grid
        ax4 = plt.subplot(gs[2, 1])
        ax4.axis('off')
        stats_text = f"""
Signal Statistics:
ADC Values:
  Mean: {np.mean(values):.1f}
  Std Dev: {np.std(values):.1f}
  Max: {np.max(values)}
  Min: {np.min(values)}
  Range: {np.max(values) - np.min(values)}

Recording:
  Duration: {timestamps[-1]:.1f} seconds
  Sampling Rate: {sample_rate:.1f} Hz
  Samples: {len(values)}
"""
        ax4.text(0, 0.5, stats_text, fontsize=8, va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        output_file = npz_file.replace('.npz', '_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nAnalysis complete! Visualization saved as: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Error details:", sys.exc_info())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 npToGraph.py <npz_file>")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    if not npz_file.endswith('.npz'):
        print("Error: File must be a .npz file")
        sys.exit(1)
        
    try:
        plot_eeg_data(npz_file)
    except Exception as e:
        print(f"Failed to process file: {e}")