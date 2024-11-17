// Arduino code (save as eeg_collection.ino)
const int EEG_PIN = A0;  // Analog pin for EEG reading
const int SAMPLE_RATE = 200;  // Sampling rate in Hz
const unsigned long SAMPLE_INTERVAL = 1000000 / SAMPLE_RATE;  // Interval in microseconds

void setup() {
  Serial.begin(115200);  // High baud rate for faster data transfer
  analogReference(DEFAULT);  // Using default 5V reference
}

void loop() {
  static unsigned long lastSampleTime = 0;
  unsigned long currentTime = micros();
  
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    // Read EEG value
    int eegValue = analogRead(EEG_PIN);
    
    // Send timestamp (ms) and EEG value
    Serial.print(millis());
    Serial.print(",");
    Serial.println(eegValue);
    
    lastSampleTime = currentTime;
  }
}