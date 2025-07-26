## Environment
1. Install python3.7, 3.8 or 3.9. 
2. Install the requirements in [requirements.txt](https://github.com/Wentao-Xu/HIST/blob/main/requirements.txt).
3. Install the quantitative investment platform [Qlib](https://github.com/microsoft/qlib) and download the data from Qlib:
	```
	# install Qlib from source
	pip install --upgrade  cython
	git clone https://github.com/microsoft/qlib.git && cd qlib
	python setup.py install

	# Download the stock features of Alpha360 from Qlib
	python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
	```