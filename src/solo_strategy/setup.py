from setuptools import setup

package_name = 'solo_strategy'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gustavo',
    maintainer_email='gustavomoura@discente.ufg.br',
    description='One-robot strategy node',
    license='MIT',
    entry_points={'console_scripts': ['solo_node = solo_strategy.solo_node:main']},
)
