from glob import glob

from setuptools import find_packages, setup

package_name = 'vision_filter'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='gus',
    maintainer_email='gustavomoura@discente.ufg.br',
    description='Per-robot EKF over SSL-Vision: filtered poses and body velocities (/vision/filtered/*).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_filter = vision_filter.node:main',
            'fake_vision = vision_filter.fake_vision:main',
            'vision_check = vision_filter.vision_check:main',
        ],
    },
)
