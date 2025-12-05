from setuptools import setup, find_packages

setup(
    name="mbd",
    author="Chaoyi Pan",
    author_email="chaoyip@andrew.cmu.edu",
    # Discover the actual package and subpackages so `import mbd` succeeds.
    packages=find_packages(include=["mbd", "mbd.*"]),
    version="0.0.1",
    install_requires=[
        "gym",
        "pandas",
        "seaborn",
        "matplotlib",
        "imageio",
        "control",
        "tqdm",
        "tyro",
        "meshcat",
        "sympy",
        "gymnax",
        "jax",
        "brax",
        "distrax",
        "gputil",
        "jaxopt",
    ],
)