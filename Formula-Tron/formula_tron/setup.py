from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'formula_tron'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
        'pyqtgraph',  # Real-time plotting for telemetry dashboard
        'casadi',     # NLP solver for MPC controller
    ],
    zip_safe=True,
    maintainer='Formula-Tron Team',
    maintainer_email='noreply@formulatron.dev',
    description='Formula-Tron: Vision-based autonomous racing for F1TENTH',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vision_controller = formula_tron.vision_controller:main',
            'control_gui = formula_tron.control_gui:main',
        ],
    },
    test_suite='pytest',
    tests_require=['pytest'],
)
