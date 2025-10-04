from setuptools import find_packages, setup
from glob import glob

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thiago',
    maintainer_email='thiago.017henrique@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = robot_control.controller:main',
            'stm32_bridge = robot_control.stm32_bridge:main',
            'imu = robot_control.imu:main',
            'velocity_pid = robot_control.velocity_pid:main'
        ],
    },
)