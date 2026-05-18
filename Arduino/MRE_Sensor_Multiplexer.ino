/*
  Up to 8 supported PCBs via PCA9548 Multiplexer
  Reads up to 8 sets of 5 magnetometer arrays (same I2C addresses)
  using the Adafruit PCA9548 8-Channel I2C Multiplexer (channels 0-7).

  At startup, each channel is probed. If no device responds, that channel
  is marked inactive and skipped during the read/print loop.

  Output format per active channel (one line per loop iteration):
    CHn,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
  where n is the mux channel number (0-7).
*/

#include <Wire.h>
#include <MLX90393.h>

#define Serial SERIAL_PORT_USBVIRTUAL

// PCA9548 Multiplexer I2C address (default: 0x70)
#define PCA9548_ADDR 0x70

// Total number of mux channels to scan
#define NUM_CHANNELS 8

// Number of sensors per channel
#define SENSORS_PER_CHANNEL 5

// Shared I2C addresses for each sensor slot across all channels
const uint8_t SENSOR_ADDR[SENSORS_PER_CHANNEL] = {
  0x0F, // Sensor 0 - Top
  0x0E, // Sensor 1 - Left
  0x18, // Sensor 2 - Middle
  0x0D, // Sensor 3 - Right
  0x0C  // Sensor 4 - Bottom
};

// Per-channel sensor objects and data buffers
MLX90393         mlx[NUM_CHANNELS][SENSORS_PER_CHANNEL];
MLX90393::txyz   data[NUM_CHANNELS][SENSORS_PER_CHANNEL];

// Tracks which channels have active (connected) sensor arrays
bool channelActive[NUM_CHANNELS];

// ── Mux helpers ──────────────────────────────────────────────────────────────

// Selects the multiplexer channel to connect to
void selectMuxChannel(uint8_t channel) {
  Wire.beginTransmission(PCA9548_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

// Closes all of the channels in the multiplexer
void disableMux() {
  Wire.beginTransmission(PCA9548_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}

// Used to determine which channel has an active sensor
bool probeI2C(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(2000000); // 2 Mbaud for fast transmission
  while (!Serial) {
    delay(5);
  }

  Wire.begin();
  Wire.setClock(400000); // Fast Mode I2C at 400 kHz
  delay(10);

  // Probe and initialize each channel
  for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
    selectMuxChannel(ch);
    delay(5);

    // A channel is considered active if at least the first sensor responds
    if (!probeI2C(SENSOR_ADDR[0])) {
      channelActive[ch] = false;
      continue;
    }

    channelActive[ch] = true;

    // Initialise every sensor on this channel
    for (uint8_t s = 0; s < SENSORS_PER_CHANNEL; s++) {
      mlx[ch][s].begin(SENSOR_ADDR[s], -1, Wire);
      mlx[ch][s].startBurst(0xF);
    }
  }

  disableMux();
}

void loop() {
  // "ST," is a frame-sync marker printed before all channel data.
  // If bytes are lost mid-stream, the partial line they corrupt will fail
  Serial.println("ST,");

  // Read all active channels first, then print — minimises latency skew
  for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
    if (!channelActive[ch]) continue;

    selectMuxChannel(ch);
    for (uint8_t s = 0; s < SENSORS_PER_CHANNEL; s++) {
      mlx[ch][s].readBurstData(data[ch][s]);
    }
  }

  disableMux();

  // Print one CSV line per active channel, prefixed with port number
  for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
    if (!channelActive[ch]) continue;

    // Port prefix
    Serial.print("CH");
    Serial.print(ch);
    Serial.print(',');

    // 15 values: x,y,z for each of the 5 sensors
    for (uint8_t s = 0; s < SENSORS_PER_CHANNEL; s++) {
      Serial.print(data[ch][s].x); Serial.print(',');
      Serial.print(data[ch][s].y); Serial.print(',');
      if (s < SENSORS_PER_CHANNEL - 1) {
        Serial.print(data[ch][s].z); Serial.print(',');
      } else {
        Serial.println(data[ch][s].z);
      }
    }
  }

  // End-of-frame marker so the Python side knows a complete frame arrived
  Serial.println("EN");
  delayMicroseconds(50);
}
