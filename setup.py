from setuptools import setup


packages = [
    'cnn_sys_ident',
]
install_requires = [
    'numpy<1.17',
    'scipy',
    # 'tensorflow==1.14',  # i.e. CPU only
    # 'tensorflow-gpu==1.14',  # i.e. GPU only
]
setup_requires = [
]
test_requires = [
]
entry_points = {
    'console_scripts': [
    ],
    'gui_scripts': [
    ]
}

setup(
    name='cnn_sys_ident',
    version='0.0.0',
    packages=packages,
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    entry_points=entry_points,
)
