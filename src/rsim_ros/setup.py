from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rsim_ros'

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
            'simulator = src.simulator:main',
            'render = src.render:main',
            'gamepad = src.gamepad:main',
        ],
    },
)
