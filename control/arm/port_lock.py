# control/arm/port_lock.py
#
# File-based lock to prevent two ArmController instances (e.g. main.py's
# dev path and joint_bridge_node.py's ROS2 path) from opening the same
# serial port simultaneously.
#
# pySerial does NOT lock the port at the OS level by default on Linux —
# two processes CAN both open /dev/ttyUSB0 at once. Neither will raise an
# error. Instead both will read/write to the same port and corrupt each
# other's data silently. This is worse than a clean crash, which is why
# this explicit lock exists.
#
# Uses fcntl.flock() — an advisory lock tied to the lock file. Released
# automatically if the process crashes or exits (the OS releases flocks
# on process death), so no manual cleanup needed in normal operation.

import fcntl
import os
import errno


class PortAlreadyInUseError(RuntimeError):
    """Raised when another process already holds the lock for this port."""
    pass


class SerialPortLock:
    """
    Advisory file lock for a serial port path.

    Usage:
        lock = SerialPortLock('/dev/ttyUSB0')
        lock.acquire()   # raises PortAlreadyInUseError if already held
        ...
        lock.release()   # or just let the process exit — OS releases it

    Lock file lives at /tmp/<sanitized_port_name>.lock — separate from the
    actual device file, since you cannot flock a character device reliably
    across all platforms, but a sidecar lock file works everywhere.
    """

    def __init__(self, port: str):
        self.port = port
        safe_name = port.replace('/', '_').strip('_')
        self.lock_path = f"/tmp/arm_serial_{safe_name}.lock"
        self._fd = None

    def acquire(self):
        """
        Attempt to acquire the lock. Raises PortAlreadyInUseError if another
        process already holds it. Safe to call multiple times from the same
        process (idempotent) since flock from the same process re-locking
        the same fd succeeds.
        """
        
        if self._fd is not None:
            return
        
        self._fd = open(self.lock_path, 'w')
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self._fd.close()
            self._fd = None
            if e.errno in (errno.EACCES, errno.EAGAIN):
                raise PortAlreadyInUseError(
                    f"Port '{self.port}' is already locked by another "
                    f"process. Two ArmController instances cannot open the "
                    f"same serial port — check whether main.py and "
                    f"joint_bridge_node.py (or two copies of either) are "
                    f"running simultaneously. Lock file: {self.lock_path}"
                )
            raise
            
            self._fd.seek(0)
            self._fd.truncate()
            self._fd.write(str(os.getpid()))
            self._fd.flush()

        # Record the PID for diagnostic purposes — visible if someone cats
        # the lock file while debugging.
        self._fd.write(str(os.getpid()))
        self._fd.flush()

    def release(self):
        """Release the lock explicitly. Safe to call even if never acquired."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            self._fd.close()
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()