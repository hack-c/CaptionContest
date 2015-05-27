from setuptools import setup, find_packages

setup(
        name='captioncontest',
        version='0.1',
        author="Charlie Hack",
        author_email="charles.t.hack@gmail.com",
        description="Library for preprocessing and visualizing The New Yorker Cartoon Department spreadsheets.",
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            'numpy',
            'pandas',
            'click',
        ],
        entry_points={
            'console_scripts': ['xls2csv=captioncontest.xls2csv:main'],
        },
)