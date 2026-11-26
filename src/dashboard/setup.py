from glob import glob

from setuptools import find_packages, setup

package_name = 'dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.yaml')),
        ('share/' + package_name + '/web', glob('web/*')),
    ],
    install_requires=['setuptools', 'websockets'],
    zip_safe=True,
    maintainer='Gustavo',
    maintainer_email='gustavomoura@discente.ufg.br',
    description='Live field dashboard for the SSL stack.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dashboard = dashboard.node:main',
        ],
    },
)
