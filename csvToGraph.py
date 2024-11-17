import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import seaborn as sns

def plot_eeg_data(csv_file):
    """
    Create comprehensive EEG visualization from CSV data
    
    Parameters:
    csv_file (str): Path to CSV file containing EEG data
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to seconds for better readability
    df['Time_Seconds'] = df['Timestamp_ms'] / 1000.0
    
    # Calculate sampling rate
    sample_rate = 1000 / np.mean(np.diff(df['Timestamp_ms']))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 0.5])
    
    # Plot 1: Raw EEG Signal
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['Time_Seconds'], df['EEG_Value'], 'b-', linewidth=0.5, alpha=0.8)
    ax1.set_title('Raw EEG Signal', fontsize=12, pad=10)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Calculate and plot spectrogram
    ax2 = fig.add_subplot(gs[1])
    frequencies, times, Sxx = signal.spectrogram(df['EEG_Value'], fs=sample_rate,
                                               nperseg=min(256, len(df)//4),
                                               noverlap=min(128, len(df)//8))
    im = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Spectrogram', fontsize=12, pad=10)
    plt.colorbar(im, ax=ax2, label='Power/Frequency (dB/Hz)')
    
    # Plot 3: Signal Statistics
    ax3 = fig.add_subplot(gs[2])
    sns.boxplot(x=df['EEG_Value'], ax=ax3, color='lightblue')
    ax3.set_title('Signal Distribution', fontsize=12, pad=10)
    ax3.set_xlabel('Amplitude')
    
    # Add signal statistics as text
    stats_text = f"""
    Signal Statistics:
    Mean: {df['EEG_Value'].mean():.2f}
    Std Dev: {df['EEG_Value'].std():.2f}
    Max: {df['EEG_Value'].max():.2f}
    Min: {df['EEG_Value'].min():.2f}
    Duration: {df['Time_Seconds'].max():.1f} seconds
    Sampling Rate: {sample_rate:.1f} Hz
    """
    plt.figtext(0.85, 0.2, stats_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Save the plot
    output_file = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Use the most recent CSV file in the current directory
        import glob
        csv_files = glob.glob('eeg_data_*.csv')
        if not csv_files:
            print("No EEG data files found!")
            sys.exit(1)
        csv_file = max(csv_files, key=os.path.getctime)
    
    print(f"Analyzing {csv_file}...")
    output_file = plot_eeg_data(csv_file)
    print(f"Analysis complete! Visualization saved as: {output_file}")