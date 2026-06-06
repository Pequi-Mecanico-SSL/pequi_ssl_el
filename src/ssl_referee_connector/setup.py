from setuptools import find_packages, setup
from glob import glob

package_name = 'ssl_referee_connector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.yaml')),
    ],
    install_requires=['setuptools', 'protobuf'],
    zip_safe=True,
    maintainer='gus',
    maintainer_email='gustavomoura@discente.ufg.br',
    description='ROS 2 connector node for SSL Game Controller (Referee) protobuf.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'referee_to_ros = ssl_referee_connector.referee_protobuf_to_ros:main'
        ],
    },
)

