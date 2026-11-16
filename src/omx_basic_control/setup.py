from setuptools import setup

package_name = 'omx_basic_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'omx_basic_control = omx_basic_control.omx_basic_control:main',
            'omx_subscriber = omx_basic_control.subcriber:main',
            'forward_kin = omx_basic_control.forward:main',
            'inverse_kin = omx_basic_control.inverse:main',
        ],
    },
)
