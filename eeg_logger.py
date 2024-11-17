import serial
import csv
from datetime import datetime
import time
import serial.tools.list_ports
import os

def play_beep(type="start"):
    """Play a beep sound"""
    try:
        if type == "start":
            # Two short beeps for start
            os.system('afplay /System/Library/Sounds/Ping.aiff')

        else:
            # One long beep for end
            os.system('afplay /System/Library/Sounds/Glass.aiff')
    except:
        print("Beep!")  # Fallback if sound doesn't work

def find_arduino_port():
    """Find the Arduino's COM port"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'usbmodem' in port.device or 'usbserial' in port.device:
            return port.device
    return None

def record_eeg_data(duration_seconds=60, sample_rate=200):
    """
    Record EEG data from Arduino and save to CSV
    
    Parameters:
    duration_seconds (int): How long to record for
    sample_rate (int): Expected samples per second
    """
    # Find Arduino port
    port = find_arduino_port()
    if not port:
        print("Arduino not found! Please check connection.")
        return
    
    # Create unique filename with timestamp
    filename = "Data_no25.npz"
    
    try:
        # Connect to Arduino
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        
        print(f"Recording data for {duration_seconds} seconds...")
        print("GET READY...")
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        print("Recording started!")
        play_beep("start")
        
        # Initialize lists for data
        timestamps = []
        values = []
        
        # Calculate end time
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Counter for progress updates
        samples_received = 0
        last_update_time = start_time
        
        # Main recording loop
        while time.time() < end_time:
            try:
                # Read line from Arduino
                line = ser.readline().decode().strip()
                if line:
                    timestamp, value = line.split(',')
                    
                    timestamps.append(float(timestamp))
                    values.append(float(value))
                    samples_received += 1
                    
                    # Progress update every second
                    current_time = time.time()
                    if current_time - last_update_time >= 1:
                        elapsed = current_time - start_time
                        remaining = duration_seconds - elapsed
                        print(f"Time remaining: {remaining:.1f}s, Samples: {samples_received}")
                        last_update_time = current_time
                        
            except Exception as e:
                print(f"Error reading data: {e}")
                continue
        
        # Convert to numpy arrays
        import numpy as np
        timestamps = np.array(timestamps)
        values = np.array(values)
        voltages = values * (5.0 / 1023.0)  # Convert to voltage
        
        # Save arrays
        np.savez(filename, 
                timestamps=timestamps,
                values=values, 
                voltages=voltages)
        
        # Final statistics
        total_time = time.time() - start_time
        actual_sample_rate = samples_received / total_time
        
        play_beep("end")  # End beep
        print("\nRecording complete!")
        print(f"Total samples: {samples_received}")
        print(f"Actual sample rate: {actual_sample_rate:.1f} Hz")
        print(f"Data saved to: {filename}")
        
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    # Record for 60 seconds at 200 Hz
    record_eeg_data(duration_seconds = 60, sample_rate=200)