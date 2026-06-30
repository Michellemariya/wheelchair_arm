/*
 * arm_firmware.ino
 * ESP32 firmware for 5-DOF servo arm with MT6701 magnetic encoders in SSI mode.
 *
 * Hardware:
 *   - 5x hobby servos on pins defined below
 *   - 1x gripper servo
 *   - 5x MT6701 encoders via SPI (SSI mode), one CS pin per encoder
 *   - UART2 communication with Raspberry Pi
 *
 * MT6701 SSI wiring (all encoders share SCLK + MISO):
 *   SCLK → GPIO 18
 *   MISO → GPIO 19   (MT6701 DO pin)
 *   MOSI → not connected (SSI is read-only)
 *   CS0  → GPIO 5    (joint 1)
 *   CS1  → GPIO 15   (joint 2)
 *   CS2  → GPIO 2    (joint 3)
 *   CS3  → GPIO 4    (joint 4)
 *   CS4  → GPIO 16   (joint 5)
 *
 * MT6701 SSI protocol:
 *   - CS low → clock 25 bits → CS high
 *   - Bits [24:11] = 14-bit angle (MSB first)
 *   - Bits [10:0]  = status/alarm bits (ignore for position)
 *   - Clock idle high (CPOL=1), sample on falling edge (CPHA=1) → SPI Mode 3
 *   - Max clock: 1 MHz recommended for SSI
 *
 * Packet format (Pi -> ESP32):
 *   Joint command:    [0xAA][j1_f][j2_f][j3_f][j4_f][j5_f][checksum]
 *   Velocity command: [0xAB][vx_f][vy_f][vz_f][checksum]
 *   Gripper open:     [0xAC][0x00][checksum]
 *   Gripper close:    [0xAC][0x01][checksum]
 *   Stop:             [0xAD][checksum]
 *   Home:             [0xAE][checksum]
 *
 * Packet format (ESP32 -> Pi):
 *   Feedback: [0xBB][j1_f][j2_f][j3_f][j4_f][j5_f][current_f][checksum]
 *
 * All floats: 4-byte IEEE 754, big-endian.
 * Checksum: sum of all payload bytes mod 256.
 */

#include <ESP32Servo.h>
#include <SPI.h>

// ─── Servo pins ───────────────────────────────────────────────────────────────
#define SERVO_PIN_J1   13
#define SERVO_PIN_J2   12
#define SERVO_PIN_J3   14
#define SERVO_PIN_J4   27
#define SERVO_PIN_J5   26
#define SERVO_PIN_GRIP 25

// ─── SPI / MT6701 pins ────────────────────────────────────────────────────────
#define SPI_SCLK    18
#define SPI_MISO    19
// MOSI not needed for SSI read-only

const int CS_PINS[5] = { 5, 15, 2, 4, 16 };   // one per joint

// ─── UART2 (Pi communication) ─────────────────────────────────────────────────
#define UART_RX     17
#define UART_TX     17
// NOTE: On ESP32, Serial2 default is RX=16, TX=17.
// If GPIO16 is used as CS4, reassign CS4 to another free pin (e.g. GPIO 32)
// and keep Serial2 defaults. Pinout shown above uses GPIO16 for CS4 —
// change CS_PINS[4] to 32 if you use Serial2 on default pins.
#define UART_BAUD   115200

// ─── Constants ────────────────────────────────────────────────────────────────
#define N_JOINTS          5
#define MT6701_BITS       25          // SSI frame: 25 clock pulses
#define MT6701_ANGLE_BITS 14          // top 14 bits are angle
#define MT6701_MAX        16384.0f    // 2^14

// SPI clock for MT6701 SSI — 1 MHz is safe
#define SPI_CLOCK_HZ      1000000

// Command bytes (must match arm_controller.py)
#define CMD_JOINT      0xAA
#define CMD_VELOCITY   0xAB
#define CMD_GRIPPER    0xAC
#define CMD_STOP       0xAD
#define CMD_HOME       0xAE
#define FB_JOINT       0xBB

// Servo limits (microseconds) — adjust per your servo datasheet
#define SERVO_MIN_US   500
#define SERVO_MAX_US   2500

// Joint limits in degrees — edit to match your physical arm
const float JOINT_MIN_DEG[N_JOINTS] = {  0.0,  0.0,  0.0,  0.0,  0.0 };
const float JOINT_MAX_DEG[N_JOINTS] = {180.0,180.0,180.0,180.0,180.0 };

// Home position in degrees
const float HOME_DEG[N_JOINTS] = { 90.0, 90.0, 90.0, 90.0, 90.0 };

// Gripper angles
#define GRIPPER_OPEN_DEG   30.0f
#define GRIPPER_CLOSE_DEG 150.0f

// Feedback rate
#define FEEDBACK_INTERVAL_MS  10    // 100 Hz

// ─── Globals ──────────────────────────────────────────────────────────────────
Servo servos[N_JOINTS];
Servo gripper;

float joint_angles_deg[N_JOINTS];    // last commanded angles
float encoder_angles_deg[N_JOINTS];  // live encoder readings
float gripper_current = 0.0f;

unsigned long last_feedback_ms = 0;

SPIClass spi_bus(VSPI);   // use VSPI peripheral

// ─── MT6701 SSI read ──────────────────────────────────────────────────────────
/*
 * MT6701 SSI protocol:
 *   CS low → send 25 clock pulses → CS high
 *   We read 4 bytes (32 bits) via SPI, then extract top 14 bits.
 *
 *   Bit layout of the 25-bit SSI frame (MSB first):
 *     Bits 24..11 → 14-bit angle
 *     Bits 10..0  → status flags (magnetic field strength, alarm)
 *
 *   SPI Mode 3 (CPOL=1, CPHA=1): clock idle high, data sampled on falling edge.
 */
float read_encoder_deg(int joint_index) {
    int cs = CS_PINS[joint_index];

    // Begin transaction: Mode 3, 1 MHz
    spi_bus.beginTransaction(SPISettings(SPI_CLOCK_HZ, MSBFIRST, SPI_MODE3));

    digitalWrite(cs, LOW);
    delayMicroseconds(1);   // CS setup time (MT6701 needs >100ns)

    // Read 4 bytes (32 bits) — we only need 25 bits
    // MT6701 sends MSB first; top 14 bits are angle
    uint8_t b0 = spi_bus.transfer(0x00);
    uint8_t b1 = spi_bus.transfer(0x00);
    uint8_t b2 = spi_bus.transfer(0x00);
    uint8_t b3 = spi_bus.transfer(0x00);   // extra byte to clock out remaining bits

    delayMicroseconds(1);
    digitalWrite(cs, HIGH);

    spi_bus.endTransaction();

    // Reconstruct 32-bit word and extract top 14 bits
    // SSI frame: bit24=MSB of angle, bit11=LSB of angle
    uint32_t raw32 = ((uint32_t)b0 << 24) |
                     ((uint32_t)b1 << 16) |
                     ((uint32_t)b2 <<  8) |
                      (uint32_t)b3;

    // Shift right by (32 - 14) = 18 to get top 14 bits
    uint16_t angle_raw = (raw32 >> 18) & 0x3FFF;

    // Convert to degrees: 16384 counts = 360 degrees
    return ((float)angle_raw / MT6701_MAX) * 360.0f;
}

void read_all_encoders() {
    for (int i = 0; i < N_JOINTS; i++) {
        encoder_angles_deg[i] = read_encoder_deg(i);
    }
}

// ─── Servo control ────────────────────────────────────────────────────────────
void write_servo_deg(int joint, float deg) {
    deg = constrain(deg, JOINT_MIN_DEG[joint], JOINT_MAX_DEG[joint]);
    joint_angles_deg[joint] = deg;
    servos[joint].write((int)deg);
}

void move_to_home() {
    for (int i = 0; i < N_JOINTS; i++) {
        write_servo_deg(i, HOME_DEG[i]);
    }
    gripper.write((int)GRIPPER_OPEN_DEG);
    Serial.println("Moved to home position");
}

// ─── Checksum ─────────────────────────────────────────────────────────────────
uint8_t compute_checksum(uint8_t* data, int len) {
    uint32_t sum = 0;
    for (int i = 0; i < len; i++) sum += data[i];
    return (uint8_t)(sum % 256);
}

// ─── Float byte conversion (big-endian ↔ little-endian) ──────────────────────
float bytes_to_float_be(uint8_t* b) {
    // ESP32 is little-endian; incoming bytes are big-endian — swap
    uint8_t tmp[4] = { b[3], b[2], b[1], b[0] };
    float val;
    memcpy(&val, tmp, 4);
    return val;
}

void float_to_bytes_be(float val, uint8_t* out) {
    uint8_t tmp[4];
    memcpy(tmp, &val, 4);
    // Reverse byte order for big-endian output
    out[0] = tmp[3];
    out[1] = tmp[2];
    out[2] = tmp[1];
    out[3] = tmp[0];
}

// ─── Feedback packet ──────────────────────────────────────────────────────────
void send_feedback() {
    /*
     * [0xBB][j1_rad_f][j2_rad_f][j3_rad_f][j4_rad_f][j5_rad_f][current_f][cs]
     * Total: 1 + 6*4 + 1 = 26 bytes
     * Sends real encoder angles in radians.
     */
    const int PAYLOAD_LEN = 1 + N_JOINTS * 4 + 4;
    uint8_t payload[PAYLOAD_LEN];
    int idx = 0;

    payload[idx++] = FB_JOINT;

    for (int i = 0; i < N_JOINTS; i++) {
        float rad = encoder_angles_deg[i] * (float)M_PI / 180.0f;
        float_to_bytes_be(rad, &payload[idx]);
        idx += 4;
    }

    float_to_bytes_be(gripper_current, &payload[idx]);
    idx += 4;

    uint8_t cs = compute_checksum(payload, idx);
    Serial2.write(payload, idx);
    Serial2.write(cs);
}

// ─── Command handlers ─────────────────────────────────────────────────────────
void handle_joint_command(uint8_t* data) {
    // data = [j1_f][j2_f][j3_f][j4_f][j5_f][checksum] = 21 bytes
    int payload_len = N_JOINTS * 4;

    uint8_t cs_buf[1 + payload_len];
    cs_buf[0] = CMD_JOINT;
    memcpy(&cs_buf[1], data, payload_len);

    if (compute_checksum(cs_buf, 1 + payload_len) != data[payload_len]) {
        Serial.println("Joint cmd: checksum error");
        return;
    }

    for (int i = 0; i < N_JOINTS; i++) {
        float rad = bytes_to_float_be(&data[i * 4]);
        float deg = rad * 180.0f / (float)M_PI;
        write_servo_deg(i, deg);
    }
}

void handle_velocity_command(uint8_t* data) {
    /*
     * data = [vx_f][vy_f][vz_f][checksum] = 13 bytes
     *
     * Velocity is in OpenCV camera frame: +X right, +Y down, +Z forward.
     * Current mapping is a placeholder heuristic:
     *   vx → joint 0 (base rotate)
     *   vy → joint 1 (shoulder tilt, up/down)
     *   vz → joint 2 (elbow, forward/back)
     *
     * Replace with proper Jacobian transpose once DH parameters are defined.
     */
    int payload_len = 3 * 4;

    uint8_t cs_buf[1 + payload_len];
    cs_buf[0] = CMD_VELOCITY;
    memcpy(&cs_buf[1], data, payload_len);

    if (compute_checksum(cs_buf, 1 + payload_len) != data[payload_len]) {
        Serial.println("Velocity cmd: checksum error");
        return;
    }

    float vx = bytes_to_float_be(&data[0]);
    float vy = bytes_to_float_be(&data[4]);
    float vz = bytes_to_float_be(&data[8]);

    // Scale factor: tune this empirically for your arm speed
    float scale = 10.0f;

    write_servo_deg(0, joint_angles_deg[0] + vx * scale);
    write_servo_deg(1, joint_angles_deg[1] + vy * scale);
    write_servo_deg(2, joint_angles_deg[2] + vz * scale);
}

void handle_gripper_command(uint8_t* data) {
    // data = [state][checksum] = 2 bytes
    uint8_t cs_buf[2] = { CMD_GRIPPER, data[0] };
    if (compute_checksum(cs_buf, 2) != data[1]) {
        Serial.println("Gripper cmd: checksum error");
        return;
    }

    if (data[0] == 0x00) {
        gripper.write((int)GRIPPER_OPEN_DEG);
        Serial.println("Gripper: open");
    } else {
        gripper.write((int)GRIPPER_CLOSE_DEG);
        Serial.println("Gripper: close");
    }
}

void handle_stop_command(uint8_t cs) {
    uint8_t cs_buf[1] = { CMD_STOP };
    if (compute_checksum(cs_buf, 1) != cs) {
        Serial.println("Stop cmd: checksum error");
        return;
    }
    // Servos hold last position by default — no action needed
    Serial.println("STOP");
}

void handle_home_command(uint8_t cs) {
    uint8_t cs_buf[1] = { CMD_HOME };
    if (compute_checksum(cs_buf, 1) != cs) {
        Serial.println("Home cmd: checksum error");
        return;
    }
    move_to_home();
}

// ─── UART packet reader ───────────────────────────────────────────────────────
void process_uart() {
    if (!Serial2.available()) return;

    uint8_t cmd = Serial2.read();

    switch (cmd) {

        case CMD_JOINT: {
            // 5 floats + 1 checksum = 21 bytes
            uint8_t buf[21];
            if (Serial2.readBytes(buf, 21) == 21)
                handle_joint_command(buf);
            break;
        }

        case CMD_VELOCITY: {
            // 3 floats + 1 checksum = 13 bytes
            uint8_t buf[13];
            if (Serial2.readBytes(buf, 13) == 13)
                handle_velocity_command(buf);
            break;
        }

        case CMD_GRIPPER: {
            // 1 state + 1 checksum = 2 bytes
            uint8_t buf[2];
            if (Serial2.readBytes(buf, 2) == 2)
                handle_gripper_command(buf);
            break;
        }

        case CMD_STOP: {
            uint8_t cs;
            if (Serial2.readBytes(&cs, 1) == 1)
                handle_stop_command(cs);
            break;
        }

        case CMD_HOME: {
            uint8_t cs;
            if (Serial2.readBytes(&cs, 1) == 1)
                handle_home_command(cs);
            break;
        }

        default:
            // Unknown byte — discard and resync naturally
            break;
    }
}

// ─── Gripper current ──────────────────────────────────────────────────────────
void read_gripper_current() {
    /*
     * Placeholder — returns 0.0 until a current sense resistor is wired.
     *
     * To implement: connect an INA219 or a shunt resistor to an ADC pin.
     * Example with ADC:
     *   int raw = analogRead(34);
     *   float voltage = raw * (3.3f / 4095.0f);
     *   gripper_current = voltage / 0.1f;  // 0.1 ohm shunt
     */
    gripper_current = 0.0f;
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    // USB debug serial
    Serial.begin(115200);
    Serial.println("=== ARM CONTROLLER BOOT ===");
    Serial.println("MT6701 SSI encoders | ESP32Servo | UART2");

    // UART to Pi (Serial2 default: RX=16, TX=17)
    // NOTE: if GPIO16 conflicts with CS4, change CS_PINS[4] to GPIO32
    Serial2.begin(UART_BAUD, SERIAL_8N1, 16, 17);

    // CS pins — all high (deselected) before SPI starts
    for (int i = 0; i < N_JOINTS; i++) {
        pinMode(CS_PINS[i], OUTPUT);
        digitalWrite(CS_PINS[i], HIGH);
    }

    // SPI bus init (VSPI: SCLK=18, MISO=19, MOSI=23 — MOSI unused)
    spi_bus.begin(SPI_SCLK, SPI_MISO, -1);   // -1 = no MOSI

    // Servo PWM timer allocation (ESP32Servo requirement)
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    // Attach servos
    const int servo_pins[N_JOINTS] = {
        SERVO_PIN_J1, SERVO_PIN_J2, SERVO_PIN_J3,
        SERVO_PIN_J4, SERVO_PIN_J5
    };
    for (int i = 0; i < N_JOINTS; i++) {
        servos[i].setPeriodHertz(50);
        servos[i].attach(servo_pins[i], SERVO_MIN_US, SERVO_MAX_US);
    }
    gripper.setPeriodHertz(50);
    gripper.attach(SERVO_PIN_GRIP, SERVO_MIN_US, SERVO_MAX_US);

    // Read initial encoder positions and use as starting commanded angles
    read_all_encoders();
    for (int i = 0; i < N_JOINTS; i++) {
        joint_angles_deg[i] = encoder_angles_deg[i];
        Serial.printf("Joint %d encoder: %.2f deg\n", i + 1, encoder_angles_deg[i]);
    }

    // Move to home
    move_to_home();
    delay(2000);

    Serial.println("Ready. Waiting for commands from Pi...");
}

// ─── Main loop ────────────────────────────────────────────────────────────────
void loop() {
    // Process incoming commands from Pi
    process_uart();

    // Read all MT6701 encoders via SPI
    read_all_encoders();

    // Read gripper current (placeholder)
    read_gripper_current();

    // Send encoder feedback to Pi at fixed rate
    unsigned long now = millis();
    if (now - last_feedback_ms >= FEEDBACK_INTERVAL_MS) {
        send_feedback();
        last_feedback_ms = now;
    }
}
