
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError('CMake must be installed to build the following extensions: ' +
                               ', '.join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + self.get_python_executable(),
            '-DCOMPILE_IKSolver_PYBINDING=ON'
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

    def get_python_executable(self):
        return os.path.abspath(sys.executable)

setup(
    name='IKSolver_interface_py',
    version='0.1.4',
    author='JPengyu',
    author_email='jiangpengyu@sjtu.edu.cn',
    description='IRMV IKSolver project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://irmv.sjtu.edu.cn/',
    packages=['interface'],
    ext_modules=[CMakeExtension('IKSolver_interface_py', sourcedir='.')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial 4.0 International License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
