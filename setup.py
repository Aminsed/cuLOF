#!/usr/bin/env python3
"""
Setup script for cuLOF package.
"""

import os
import sys
import platform
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Colored terminal output for better visibility of errors
try:
    RED = "\033[1;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    RESET = "\033[0;0m"
except:
    # Fallback if colors not supported
    RED = ""
    GREEN = ""
    YELLOW = ""
    BLUE = ""
    RESET = ""

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        print(f"{BLUE}Setting up CUDA-accelerated LOF extension...{RESET}")
        print(f"Python: {sys.version}")
        print(f"Platform: {platform.platform()}")
        
        # Check for CMake
        try:
            cmake_version = subprocess.check_output(['cmake', '--version']).decode('utf-8')
            print(f"{GREEN}Found CMake: {cmake_version.split('version')[1].strip()}{RESET}")
        except (OSError, subprocess.SubprocessError):
            print(f"{RED}ERROR: CMake not found!{RESET}")
            print(f"{YELLOW}CMake 3.18+ is required to build the extension.{RESET}")
            print("Please install CMake:")
            print("  • Ubuntu/Debian: apt install cmake")
            print("  • macOS: brew install cmake")
            print("  • Windows: Download from https://cmake.org/download/")
            print("  • Conda: conda install -c conda-forge cmake")
            raise RuntimeError("CMake not found. Please install CMake 3.18+ and try again.")

        # Check for CUDA
        try:
            nvcc_output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
            print(f"{GREEN}Found CUDA: {nvcc_output.strip()}{RESET}")
        except (OSError, subprocess.SubprocessError):
            print(f"{RED}ERROR: CUDA not found!{RESET}")
            print(f"{YELLOW}This package requires CUDA 11.0+ to build.{RESET}")
            print("Please verify your CUDA installation:")
            print("  1. Check that CUDA is installed: nvcc --version")
            print("  2. Make sure CUDA is in your PATH")
            print("  3. For installation instructions, visit: https://developer.nvidia.com/cuda-downloads")
            raise RuntimeError(
                "CUDA not found. This package requires CUDA 11.0+ to build. "
                "Please install CUDA toolkit from https://developer.nvidia.com/cuda-downloads "
                "and make sure nvcc is in your PATH."
            )

        # Print compiler information
        if self.compiler:
            print(f"Compiler: {self.compiler.compiler_type}")
        
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_PYTHON=ON',
            '-DBUILD_TESTS=OFF',
        ]

        # Set build type
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # CMake lets you override the generator - we need to ensure we use the same
        # one that was used to compile Python.
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        # Assuming Windows and not cross-compiling
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())

        # Make directory if it doesn't exist
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print(f"{BLUE}Building in {self.build_temp}{RESET}")
        print(f"CMake args: {' '.join(cmake_args)}")
        
        # Build the project
        try:
            print(f"{BLUE}Running CMake configuration...{RESET}")
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            
            print(f"{BLUE}Building the extension...{RESET}")
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
            
            print(f"{GREEN}Build successful! Extension will be installed to {extdir}{RESET}")
        except subprocess.CalledProcessError as e:
            print(f"{RED}Build failed with error code {e.returncode}{RESET}")
            print(f"{YELLOW}Error output:{RESET}")
            print(e.output if hasattr(e, 'output') else "No output available")
            
            # Provide more detailed troubleshooting guidance
            print(f"\n{YELLOW}Troubleshooting Tips:{RESET}")
            print("1. Verify your CUDA installation with 'nvcc --version'")
            print("2. Check that your C++ compiler is compatible with your CUDA version")
            print("3. Make sure CMake 3.18+ is installed")
            print("4. If errors persist, try installing from the GitHub repository:")
            print("   pip install git+https://github.com/Aminsed/cuLOF.git")
            print("5. For conda users, try: conda install -c conda-forge cmake cudatoolkit")
            print("   Then: pip install --no-build-isolation git+https://github.com/Aminsed/cuLOF.git")
            print("6. For detailed installation instructions, see README.md or visit the GitHub repository")
            
            raise RuntimeError(
                f"Error building the extension: {e}\n"
                "Please check that you have the required dependencies:\n"
                "- CUDA Toolkit 11.0+\n"
                "- CMake 3.18+\n"
                "- C++14 compatible compiler\n"
                "For detailed installation instructions, see https://github.com/Aminsed/cuLOF"
            ) from e

# Read the long description from README.md 
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='culof',
    version='0.1.0',
    author='Amin Sedaghat',
    author_email='amin32846@gmail.com',
    description='CUDA-accelerated Local Outlier Factor implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Aminsed/cuLOF',
    project_urls={
        'Bug Tracker': 'https://github.com/Aminsed/cuLOF/issues',
        'Documentation': 'https://github.com/Aminsed/cuLOF',
        'Source Code': 'https://github.com/Aminsed/cuLOF',
    },
    packages=find_packages(),
    ext_modules=[CMakeExtension('culof')],
    cmdclass=dict(build_ext=CMakeBuild),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.15.0',
        'matplotlib>=3.0.0',
        'scikit-learn>=0.22.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='anomaly detection, outlier detection, cuda, gpu, lof, local outlier factor',
) 