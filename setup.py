# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

from setuptools import setup

readme_dir = os.path.dirname(__file__)
readme_filename = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_filename, 'r') as f:
        readme = f.read()
except:
    logging.warn("Failed to load %s" % readme_filename)
    readme = ""

try:
    import pypandoc
    readme = pypandoc.convert(readme, to='rst', format='md')
except:
    logging.warn("Conversion of long_description from MD to RST failed")
    pass

if __name__ == '__main__':
    setup(
        name='fancyimpute',
        version="0.2.0",
        description="Matrix completion and feature imputation algorithms",
        author="Alex Rubinsteyn, Sergey Feldman",
        author_email="alex.rubinsteyn@gmail.com",
        url="https://github.com/hammerlab/fancyimpute",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        install_requires=[
            'six',
            'knnimpute',
            # need at least 1.10 for np.multi_dot
            'numpy>=1.10',
            'scipy',
            # used by NuclearNormMinimization
            'cvxpy',
            'scikit-learn>=0.17.1',
            # used by MatrixFactorization
            'downhill',
            'climate',
            'theano',
        ],
        long_description=readme,
        packages=['fancyimpute'],
    )
