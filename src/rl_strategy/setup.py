from setuptools import find_packages, setup

package_name = 'rl_strategy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/rl_strategy_launch.yaml', 'launch/rl_strategy_solo_launch.yaml']),
        # Portable weights export of the deployed checkpoint (see vendored/VENDOR.md)
        ('share/' + package_name + '/weights',
            ['weights/policy_state.portable.npz']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=False,
    maintainer='gus',
    maintainer_email='gustavomoura@discente.ufg.br',
    description=(
        "RL strategy: ROS wrapper around the inference code vendored from the team's RL repository (Pequi-Mecanico-SSL/RL, torch). Converts real-world SI vision poses into the policy's training frame and policy actions back into body-frame velocity commands."
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_strategy = rl_strategy.node:main',
        ],
    },
)
