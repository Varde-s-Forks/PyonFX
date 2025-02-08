from pathlib import Path
from typing import Any, Dict

from setuptools import setup

meta: Dict[str, Any] = {}

exec(Path('pyonfx/_metadata.py').read_text(), meta := dict[str, str]())

with open('requirements.txt', encoding='utf-8') as fh:
    reqs = fh.readlines()

with open('requirements-dev.txt', encoding='utf-8') as fh:
    reqs_dev = fh.readlines()

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pyonfx',
    url='https://github.com/CoffeeStraw/PyonFX/',
    project_urls={
        'Documentation': 'http://pyonfx.rtfd.io/',
        'Source': 'https://github.com/CoffeeStraw/PyonFX/',
        'Tracker': 'https://github.com/CoffeeStraw/PyonFX/issues/',
    },
    author='Antonio Strippoli',
    author_email='clarantonio98@gmail.com',
    description='An easy way to create KFX (Karaoke Effects) and complex typesetting using the ASS format (Advanced Substation Alpha).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=meta['__version__'],
    packages=['pyonfx', 'pyonfx.geometry', 'pyonfx.font'],
    package_data={
        'pyonfx': ['py.typed'],
    },
    python_requires='>=3.10',
    install_requires=reqs,
    extras_require={'dev': reqs_dev},
    keywords='typesetting ass subtitle aegisub karaoke kfx advanced-substation-alpha karaoke-effect',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
    ],
    license='GNU LGPL 3.0 or later',
)
