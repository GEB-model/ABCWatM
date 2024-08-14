from setuptools import setup
from pathlib import Path
from cwatm import __version__

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
    packages=[
        "cwatm",
        "cwatm.modules",
        "cwatm.modules.groundwater",
        "cwatm.modules.routing",
    ],
    package_data={
        "cwatm.modules": [
            "groundwater/mac/libmf6.dylib",
            "groundwater/linux/libmf6.so",
            "groundwater/windows/mf6.dll",
        ],
    },
    zip_safe=True,
    install_requires=[
        "numpy",
        "flopy",
        "numba",
        "xmipy>=1.4",
        "scipy",
    ],
)
