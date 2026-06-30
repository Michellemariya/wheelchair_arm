from control.arm.arm_controller import ArmController
import time

arm = ArmController()
try:
    while True:
        print(arm.get_joint_states())
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    arm.shutdown()
