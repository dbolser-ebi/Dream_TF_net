from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules=[
    Extension("sequence",
              sources=["sequence.pyx"],
              libraries=["m"] # Unix-like specific
    )
]

setup(
  name = "sequence",
  include_dirs=[numpy.get_include()],
  ext_modules = cythonize(ext_modules)
)