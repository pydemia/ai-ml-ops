
from setuptools import setup, find_packages


long_desc = """
This is made for some specific environment.
This contains ...
"""

setup(name='pre_coco',
        version='0.1',
        description='description',
        long_description=long_desc,
        url='http://github.com/pydemia/',
        author='Young Ju Kim',
        author_email='pydemia@gmail.com',
        license='MIT License',
        classifiers=[
                # How Mature: 3 - Alpha, 4 - Beta, 5 - Production/Stable
                'Development Status :: 3 - Alpha',
                'Programming Language :: Python :: 3.7'
                ],
        packages=find_packages(exclude=['contrib', 'docs', 'tests']),
        install_requires=[
            "pycocotools @ git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI",
        ],
        zip_safe=False)

