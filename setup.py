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
import re

from setuptools import setup

package_name = "fancyimpute"


readme_dir = os.path.dirname(__file__)
readme_filename = os.path.join(readme_dir, "README.md")

try:
    with open(readme_filename, "r") as f:
        readme_markdown = f.read()
except:
    logging.warn("Failed to load %s" % readme_filename)
    readme_markdown = ""

with open("%s/__init__.py" % package_name, "r") as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

if __name__ == "__main__":
    setup(
        name=package_name,
        version=version,
        description="Matrix completion and feature imputation algorithms",
        author="Alex Rubinsteyn, Sergey Feldman",
        author_email="alex.rubinsteyn@gmail.com",
        url="https://github.com/iskandr/%s" % package_name,
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Operating System :: OS Independent",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
        ],
        install_requires=[
            "knnimpute>=0.1.0",
            "scikit-learn>=0.24.2",
            # used by NuclearNormMinimization
            "cvxpy==1.1.13",
            "cvxopt",
            "numpy==1.19.5",  # tensorflow is harsh about numpy version
            # used by MatrixFactorization
            "keras>=2.4.3",
            "tensorflow>=2.5.1",
        ],
        long_description=readme_markdown,
        long_description_content_type="text/markdown",
        packages=[package_name],
    )
