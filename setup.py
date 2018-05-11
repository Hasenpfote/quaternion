#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup


if __name__ == '__main__':
    setup(
        name='quaternion',
        #packages = ['quaternion'],
        version='0.9.0',
        description='This package provides a class for manipulating quaternion objects.',
        author='Hasenpfote',
        author_email='Hasenpfote36@gmail.com',
        #url='',
        #download_url='',
        #keywords=['',],
        classifiers=[
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Development Status :: 4 - Beta',
            'Environment :: Other Environment',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        long_description='''''',
        install_requires=[
            'numpy',
        ],
    )