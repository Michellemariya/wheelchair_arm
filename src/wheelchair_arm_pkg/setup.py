from setuptools import setup, find_packages

package_name = 'wheelchair_arm_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='ROS2 nodes for wheelchair assistive arm',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'joint_bridge_node = wheelchair_arm_pkg.joint_bridge_node:main',
            'gripper_tracker_node = wheelchair_arm_pkg.gripper_tracker_node:main',
        ],
    },
)
