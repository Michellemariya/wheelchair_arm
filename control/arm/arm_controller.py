# control/arm/arm_controller.py
#Implementation on ESP32

# control/arm/arm_controller.py

import serial
import struct
import threading
import time
import numpy as np
import yaml
from pathlib import Path
from control.arm.kinematics import ArmKinematics
from control.arm.port_lock import SerialPortLock, PortAlreadyInUseError


class ArmController:
    """
    Software interface to the ESP32 arm controller via UART.

    Packet format (Pi -> ESP32):
        Joint command:     [0xAA][j1_f][j2_f][j3_f][j4_f][j5_f][checksum]
        Velocity command: [0xAB][vx_f][vy_f][vz_f][checksum]
        Gripper open:     [0xAC][0x00][checksum]
        Gripper close:    [0xAC][0x01][checksum]
        Stop:             [0xAD][checksum]
        Home:             [0xAE][checksum]

    Packet format (ESP32 -> Pi):
        Feedback: [0xBB][j1_f][j2_f][j3_f][j4_f][j5_f][current_f][checksum]

    All floats : 4-byte IEEE 754, big-endian.
    Checksum   : sum of DATA bytes only (NOT including header byte), mod 256.
                 This matches standard ESP32 firmware convention.
    """

    CMD_JOINT    = 0xAA
    CMD_VELOCITY = 0xAB
    CMD_GRIPPER  = 0xAC
    CMD_STOP     = 0xAD
    CMD_HOME     = 0xAE
    FB_JOINT     = 0xBB

    # Feedback packet structure
    _FB_N_FLOATS    = 7                         # j1..j6 + gripper_current
    _FB_DATA_SIZE   = _FB_N_FLOATS * 4           # 24 bytes
    _FB_PACKET_SIZE = 1 + _FB_DATA_SIZE + 1    # header + data + checksum = 26

    def __init__(self, config_path: str = 'configs/config.yaml'):
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Master config not found at '{config_path}'")

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        arm_cfg = cfg['arm']
        self.port         = arm_cfg['port']
        self.baud         = arm_cfg['baud']
        self.n_joints     = arm_cfg['num_joints']
        self.home_pos     = np.array(arm_cfg['home_position'], dtype=np.float64)
        self.grasp_current_threshold = float(arm_cfg['grasp_current_threshold'])
        self.comm_timeout = arm_cfg.get('comm_timeout_seconds', 0.5)

        limits = arm_cfg.get('joint_limits_rad', None)
        if limits:
            self.joint_limits = np.array(limits, dtype=np.float64)
        else:
            self.joint_limits = None
            print("[WARNING] No joint limits configured — unsafe to command angles.")

        # Thread-safe state variables
        self.joint_states        = np.zeros(self.n_joints, dtype=np.float64)
        self.gripper_current    = 0.0
        self.last_feedback_time = None

        # Synchronization Locks
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._conn_lock  = threading.Lock()  # FIXED: Protects connection states during thread transitions

        self._reader_thread = None
        self._running       = False
        self.ser            = None

        self._port_lock = SerialPortLock(self.port)
        
        # Initialize connection
        self._connect_serial()

        self.kinematics = ArmKinematics()

        # Replace with your actual encoder zeros later
        self.JOINT_ZERO_DEG = np.array([
            0,
            0,
            0,
            0,
            0,
            0
        ], dtype=np.float64)

        # Replace with actual signs later
        self.JOINT_SIGN = np.array([
            1,
            1,
            1,
            1,
            1,
            1
        ], dtype=np.float64)



    # ------------------------------------------------------------------ #
    #  Connection Management                                             #
    # ------------------------------------------------------------------ #

    def _connect_serial(self):
        """Open serial port and start reader thread. Safe to call on reconnect."""
        with self._conn_lock:
            # If a previous reader thread is alive, safely shut it down first
            if self._reader_thread is not None and self._reader_thread.is_alive():
                with self._state_lock:
                    self._running = False
                self._reader_thread.join(timeout=2.0)

            try:
                self._port_lock.acquire()   # raises PortAlreadyInUseError if held
                self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                
                with self._state_lock:
                    self._running = True
                    
                self._reader_thread = threading.Thread(
                    target=self._read_loop, daemon=True
                )
                self._reader_thread.start()
                print(f"[INFO] ESP32 connected on {self.port} at {self.baud} baud.")
            
            except PortAlreadyInUseError as e:
              # Do NOT silently fall back to simulation mode here — a port
               # conflict is a configuration error the developer needs to see
               # and fix, not something to mask as "no hardware attached."
               print(f"[FATAL] {e}")
               raise

            except serial.SerialException as e:
                print(f"[WARNING] Could not connect to ESP32: {e}")
                print("[INFO] Running in simulation mode.")
                self._port_lock.release()    # release lock since we're not using the port
                self.ser = None
                with self._state_lock:
                    self._running = False

    def _handle_disconnect(self):
        """Gracefully tear down serial link. Safe to call from any thread."""
        with self._state_lock:
            if not self._running:
                return         # Already shut down or handled
            self._running = False

        print("[WARNING] Serial link dropped. Closing port.")
        
        # Thread-safe port closure using the write lock interface
        with self._write_lock:
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

        self._port_lock.release()

    # ------------------------------------------------------------------ #
    #  Packet Helpers                                                    #
    # ------------------------------------------------------------------ #

    def _checksum(self, data_bytes) -> int:
        """Checksum = sum of DATA bytes only, mod 256."""
        return sum(data_bytes) % 256

    def _send(self, payload_bytes: list):
        """Thread-safe packet writer. payload_bytes includes command byte + data."""
        with self._write_lock:
            if self.ser is None:
                return   # Simulation mode — drop silently

            cs = self._checksum(payload_bytes)
            packet = bytes(payload_bytes) + bytes([cs])

            try:
                self.ser.write(packet)
                self.ser.flush()
            except serial.SerialException as e:
                print(f"[ERROR] Serial write failed: {e}")
                # Drop out of lock before calling handler to prevent deadlocks
                threading.Thread(target=self._handle_disconnect, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Arm Commands                                                      #
    # ------------------------------------------------------------------ #

    def send_joint_angles(self, angles):
        """Command arm to joint angles (radians) with boundary limits validation."""
        angles = np.array(angles, dtype=np.float64)

        if len(angles) != self.n_joints:
            raise ValueError(
                f"Expected {self.n_joints} joint angles, got {len(angles)}."
            )

        if self.joint_limits is not None:
            clipped = np.clip(
                angles, self.joint_limits[:, 0], self.joint_limits[:, 1]
            )
            if not np.allclose(clipped, angles, atol=1e-3):
                print(f"[WARNING] Angles clipped to joint limits. "
                      f"Requested: {angles.round(4)}, Sent: {clipped.round(4)}")
            angles = clipped

        payload = [self.CMD_JOINT]
        for a in angles:
            payload.extend(struct.pack('>f', float(a)))
        self._send(payload)

    def send_velocity(self, velocity_xyz):
        """Send Cartesian velocity (m/s) in ARM BASE FRAME (+Z up)."""
        if not self.is_comms_healthy():
            print("[WARNING] Comms unhealthy — velocity command dropped.")
            return

        vx, vy, vz = [float(v) for v in velocity_xyz[:3]]
        payload = [self.CMD_VELOCITY]
        for v in (vx, vy, vz):
            payload.extend(struct.pack('>f', v))
        self._send(payload)

    def open_gripper(self):
        self._send([self.CMD_GRIPPER, 0x00])

    def close_gripper(self):
        self._send([self.CMD_GRIPPER, 0x01])

    def stop(self):
        self._send([self.CMD_STOP])

    def move_home(self, timeout: float = 8.0, convergence_tol: float = 0.05):
        """Command arm to home position and wait for joint convergence."""
        self._send([self.CMD_HOME])

        t0 = time.time()
        while time.time() - t0 < timeout:
            current = self.get_joint_states()
            if np.allclose(current, self.home_pos, atol=convergence_tol):
                print("[INFO] Arm reached home position.")
                return
            time.sleep(0.05)

        print(f"[WARNING] move_home() timed out after {timeout}s. "
              "Arm may not have reached home position.")

    def lift(self, height_m: float = 0.1, speed: float = 0.05, timeout: float = 10.0):
        """Move arm upward in base frame Z by height_m metres."""
        try:
            start_pose = self.forward_kinematics(self.get_joint_states())
            start_z    = start_pose[2, 3]
            target_z   = start_z + height_m

            # Base frame coordinate mapping: +Z is up
            self.send_velocity(np.array([0.0, 0.0, float(speed)], dtype=np.float64))
            t0 = time.time()

            while time.time() - t0 < timeout:
                current_pose = self.forward_kinematics(self.get_joint_states())
                if current_pose[2, 3] >= target_z:
                    break
                time.sleep(0.01)

            self.stop()

        except NotImplementedError:
            # Safe Fallback Open-loop trajectory tracker
            duration = height_m / speed
            self.send_velocity(np.array([0.0, 0.0, float(speed)], dtype=np.float64))
            time.sleep(duration)
            self.stop()

    # ------------------------------------------------------------------ #
    #  State Accessors                                                   #
    # ------------------------------------------------------------------ #

    def get_joint_states(self) -> np.ndarray:
    
        raw_deg = self.read_raw_encoders_deg()

        q_deg = (
            self.JOINT_SIGN *
            (raw_deg - self.JOINT_ZERO_DEG)
        )

        q_rad = np.deg2rad(q_deg)

        return q_rad

    def get_gripper_current(self) -> float:
        with self._state_lock:
            return self.gripper_current

    def grasp_detected(self) -> bool:
        with self._state_lock:
            return self.gripper_current > self.grasp_current_threshold

    def is_comms_healthy(self) -> bool:
        with self._state_lock:
            if self.ser is None:
                return False   # Simulation mode yields explicit false tracking boundaries
            if self.last_feedback_time is None:
                return False
            return (time.time() - self.last_feedback_time) < self.comm_timeout

    def is_sim_mode(self) -> bool:
        with self._write_lock:
            return self.ser is None

    # ------------------------------------------------------------------ #
    #  Reader Thread Loop                                               #
    # ------------------------------------------------------------------ #

    def _read_loop(self):
        """Continuous byte-stream parser. Scans for FB_JOINT header without block gating."""
        DATA_SIZE = self._FB_DATA_SIZE   # 24 bytes

        while True:
            with self._state_lock:
                if not self._running:
                    break

            try:
                # Local reference capture to guarantee thread-safe checks
                with self._write_lock:
                    local_ser = self.ser
                    if local_ser is None or not local_ser.is_open:
                        local_ser = None

                if local_ser is None:
                    time.sleep(0.1)
                    continue

                # Scan for header byte (one byte at a time)
                header = local_ser.read(1)
                if not header:
                    continue
                if header[0] != self.FB_JOINT:
                    continue   # Shift buffer window forward, keep scanning

                # Read payload data frame safely
                data_bytes = local_ser.read(DATA_SIZE)
                if len(data_bytes) != DATA_SIZE:
                    print("[WARNING] Incomplete feedback packet payload — discarding.")
                    continue

                cs_byte = local_ser.read(1)
                if not cs_byte:
                    continue

                # Verify checksum over data frame exclusively
                computed_cs = self._checksum(list(data_bytes))
                received_cs = cs_byte[0]

                if computed_cs != received_cs:
                    print(f"[WARNING] Checksum mismatch: computed={computed_cs}, received={received_cs}.")
                    continue

                # Unpack big-endian IEEE 754 float values
                values = struct.unpack('>fffffff', data_bytes)

                with self._state_lock:
                    self.joint_states    = np.array(values[:6], dtype=np.float64)
                    self.gripper_current = float(values[6])
                    self.last_feedback_time = time.time()

            except serial.SerialException as e:
                print(f"[ERROR] Serial error in read loop: {e}")
                self._handle_disconnect()
                break
            except struct.error as e:
                print(f"[ERROR] Struct unpack error: {e}")
                continue

    # ------------------------------------------------------------------ #
    #  Kinematics                                                       #
    # ------------------------------------------------------------------ #

    # --------------------------------------------------
    # Replace with MT6701 reading code
    # --------------------------------------------------

    def read_raw_encoders_deg(self):
        """
    Temporary implementation.

    Right now the ESP32 already sends joint angles
    through the feedback packet.

    Later, if raw MT6701 values are exposed separately,
    replace this function.
    """

        with self._state_lock:
            return np.rad2deg(
                self.joint_states.copy()
            )
    
    def forward_kinematics(
        self,
        joint_angles: np.ndarray
    ) -> np.ndarray:

        return self.kinematics.forward_kinematics(
            joint_angles
        )
    
    def get_end_effector_pose(self):
    
        q = self.get_joint_states()

        return self.forward_kinematics(q)

    # ------------------------------------------------------------------ #
    #  Shutdown                                                         #
    # ------------------------------------------------------------------ #

    def shutdown(self):
        """Clean shutdown: stop motion, close serial, join reader thread."""
        try:
            self.stop()
        except Exception:
            pass

        with self._state_lock:
            self._running = False

        # Cleanly join the background reader thread block
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        # Force structural link drop closure safety mapping metrics
        with self._write_lock:
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
        
        self._port_lock.release()

        print("[INFO] ArmController shutdown complete.")
#Implementation on STM32
'''
import serial
import struct
import threading
import time
import numpy as np
import yaml

class ArmController:
    """
    Software interface to the physical arm via UART to STM32.
    
    Packet format (host -> STM32):
        Joint command:    [0xAA][j1_f][j2_f][j3_f][j4_f][j5_f][checksum]
        Velocity command: [0xAB][vx_f][vy_f][vz_f][checksum]
        Gripper open:     [0xAC][0x00][checksum]
        Gripper close:    [0xAC][0x01][checksum]
        Stop:             [0xAD][checksum]
        Home:             [0xAE][checksum]
    
    Packet format (STM32 -> host):
        Feedback: [0xBB][j1_f][j2_f][j3_f][j4_f][j5_f][current_f][checksum]
    
    All floats are 4-byte IEEE 754, big-endian.
    Checksum: sum of all payload bytes mod 256.
    """
    
    # Command bytes
    CMD_JOINT    = 0xAA
    CMD_VELOCITY = 0xAB
    CMD_GRIPPER  = 0xAC
    CMD_STOP     = 0xAD
    CMD_HOME     = 0xAE
    FB_JOINT     = 0xBB
    
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        limits = arm_cfg.get('joint_limits_rad', None)
        if limits:
            self.joint_limits = np.array(limits)    # shape (n_joints, 2)
        else:
            self.joint_limits = None
            print("WARNING: No joint limits configured — unsafe to command angles")
        
        arm_cfg = cfg['arm']
        self.port      = arm_cfg['port']
        self.baud      = arm_cfg['baud']
        self.n_joints  = arm_cfg['num_joints']
        self.home_pos  = np.array(arm_cfg['home_position'])
        self.grasp_current_threshold = arm_cfg['grasp_current_threshold']
        
        # State
        self.joint_states = np.zeros(self.n_joints)
        self.gripper_current = 0.0
        self._running = False
        self._lock = threading.Lock()

        self.last_feedback_time = None      # None until first packet received
        self.comm_timeout = arm_cfg.get('comm_timeout_seconds', 0.5)
        
        # Connect
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self._running = True
            self._reader_thread = threading.Thread(         #with thread, vision runs while UART continuously updates
                target=self._read_loop, daemon=True         #daemon=True ensures the thread automatically terminates when your main program exits
            )
            self._reader_thread.start()
            print(f"Arm connected on {self.port} at {self.baud} baud")
        except serial.SerialException as e:
            print(f"WARNING: Could not connect to arm: {e}")
            print("Running in simulation mode (no serial output)")
            self.ser = None
    
    def _checksum(self, data_bytes):
        return sum(data_bytes) % 256
    
    def _send(self, payload_bytes):
        if self.ser is None:
            return
        cs = self._checksum(payload_bytes)
        packet = bytes(payload_bytes) + bytes([cs])
        try:
            self.ser.write(packet)
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
    
    def send_joint_angles(self, angles):
        """
        Command arm to specific joint angles (radians).
        Clips to joint limits defined in config before sending.
        """
        angles = np.array(angles[:self.n_joints], dtype=np.float32)

        if self.joint_limits is not None:
            clipped = np.clip(angles,
                            self.joint_limits[:, 0],
                            self.joint_limits[:, 1])
            if not np.allclose(clipped, angles, atol=1e-3):
                print(f"WARNING: Joint angles clipped to limits. "
                    f"Requested: {angles}, Sent: {clipped}")
            angles = clipped

        payload = [self.CMD_JOINT]
        for a in angles:
            payload.extend(struct.pack('>f', float(a)))
        self._send(payload)
    
    def send_velocity(self, velocity_xyz):
        """
        Send Cartesian velocity command (m/s).
        velocity_xyz: [vx, vy, vz] in camera frame.
        Velocity is in OpenCV camera frame (+X right, +Y down, +Z forward).
        Caller must have transformed from base frame if needed.
        """
        if not self.is_comms_healthy():
            print("WARNING: Arm comms unhealthy — velocity command blocked")
            return
        vx, vy, vz = [float(v) for v in velocity_xyz[:3]]
        payload = [self.CMD_VELOCITY]
        payload.extend(struct.pack('>f', vx))
        payload.extend(struct.pack('>f', vy))
        payload.extend(struct.pack('>f', vz))
        self._send(payload)
    
    def stop(self):
        """Stop all motion immediately."""
        self._send([self.CMD_STOP])
    
    def move_home(self):
        """Move to home configuration."""
        self._send([self.CMD_HOME])
        time.sleep(3.0)  # wait for motion to complete
    
    def open_gripper(self):
        self._send([self.CMD_GRIPPER, 0x00])
    
    def close_gripper(self):
        self._send([self.CMD_GRIPPER, 0x01])
    
    def move_to_position(self, position_base_frame, wait=True, timeout=5.0):
        """
        Move end effector to Cartesian position in base frame.
        Computes IK internally on STM32, or here if Jacobian available.
        """
        # For now: encode as velocity-based move toward target
        # More complete: compute IK and send joint angles
        # This is a placeholder — implement IK appropriate to your arm geometry
        pass
    
    def lift(self, height_m=0.1, speed=0.05, timeout=10.0):
        """
    Move arm upward by height_m meters at speed m/s.
    Uses joint feedback to estimate progress via FK.
    Falls back to time-based if FK not implemented.
    """
        # Try closed-loop first
        try:
            start_joints = self.get_joint_states()
            start_pose   = self.forward_kinematics(start_joints)
            start_z      = start_pose[2, 3]     # Z in base frame
            target_z     = start_z + height_m

            self.send_velocity(np.array([0, -speed, 0]))
            t0 = time.time()

            while time.time() - t0 < timeout:
                current_pose = self.forward_kinematics(self.get_joint_states())
                if current_pose[2, 3] >= target_z:
                    break
                time.sleep(0.01)

            self.stop()

        except NotImplementedError:
            # FK not available — fall back to time-based
            duration = height_m / speed
            self.send_velocity(np.array([0, -speed, 0]))
            time.sleep(duration)
            self.stop()
    
    def get_joint_states(self):
        with self._lock:
            return self.joint_states.copy()
    
    def get_gripper_current(self):
        with self._lock:
            return self.gripper_current
    
    def grasp_detected(self):
        """Returns True if gripper current indicates contact."""
        return self.gripper_current > self.grasp_current_threshold
    
    def is_comms_healthy(self):
        """Returns False if no feedback received within comm_timeout seconds."""
        if self.last_feedback_time is None:
            return False    # never received anything
        return (time.time() - self.last_feedback_time) < self.comm_timeout
    
    def _read_loop(self):
        """
        Background thread: continuously reads feedback from STM32.
        Expected packet: [0xBB][5 joint floats][1 current float][checksum]
        Total: 1 + 6*4 + 1 = 26 bytes
        """
        PACKET_SIZE = 26  # 1 header + 24 data + 1 checksum
        
        while self._running:
            try:
                if self.ser.in_waiting >= PACKET_SIZE:
                    # Sync to header byte
                    byte = self.ser.read(1)         #read first byte
                    if byte != bytes([self.FB_JOINT]):          #if byte!=0xBB, skip
                        continue
                    
                    data = self.ser.read(PACKET_SIZE - 1)
                    
                    if len(data) != PACKET_SIZE - 1:
                        continue            #incomplete or oversized - discard
                    
                    # Verify checksum
                    payload = data[:-1]
                    received_cs = data[-1]
                    computed_cs = self._checksum(
                        [self.FB_JOINT] + list(payload)
                    )
                    
                    if received_cs != computed_cs:
                        continue  # corrupted packet
                    
                    # Parse 5 joint angles + 1 current value
                    values = struct.unpack('>fffffff', payload[:24])
                    
                    with self._lock:
                        self.joint_states = np.array(values[:6])
                        self.gripper_current = values[5]
                        self.last_feedback_time = time.time()
            
            except (serial.SerialException, struct.error):
                pass
            
            time.sleep(0.001)  # 1kHz read loop
    
    def forward_kinematics(self, joint_angles):
        """
    Compute end effector 4x4 transform from joint angles.
    
    TODO: implement DH parameter FK for your specific arm geometry.
    Required for hand-eye calibration and closed-loop lift.
    Until implemented, hand-eye calibration will produce wrong results.
    """
    raise NotImplementedError(
        "forward_kinematics() not implemented. "
        "Add DH parameters for your arm before using hand-eye calibration."
    )
    
    def shutdown(self):
        self.stop()
        self._running = False
        if self.ser:
            time.sleep(0.1)
            self.ser.close()
'''