from setuptools import setup
from pathlib import Path
from cwatm import __version__, __author__, __email__

setup(
    name="ABCWatM",
    version=__version__,
    description="The Community Water Model: An open source hydrological model - adapted version for coupling to ABM",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    license="GPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/GEB-model/ABCWatM",
    author=__author__,
    author_email=__email__,
    packages=[
        "cwatm",
        "cwatm.modules",
        "cwatm.modules.groundwater_modflow",
        "cwatm.modules.routing_reservoirs",
    ],
    package_data={
        "cwatm.modules": [
            "groundwater_modflow/mac/libmf6.dylib",
            "groundwater_modflow/linux/libmf6.so",
            "groundwater_modflow/windows/mf6.dll",
        ],
    },
    zip_safe=True,
    install_requires=[
        "numpy",
        "flopy",
        "numba",
        "xmipy>=1.4",
    ],
)
