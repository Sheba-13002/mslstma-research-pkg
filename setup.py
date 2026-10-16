from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("mslstma_core.model", ["mslstma_core/model.py"]),
    Extension("mslstm_core.model", ["mslstm_core/model.py"]),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)