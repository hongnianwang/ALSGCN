# ALSGCN

This repository supports the development and validation of the paper: "An Attention-based Long and Short-term Graph Convolutional Network for Stock Recommendation".

## Environment and Dataset

1. Install python3.7, 3.8 or 3.9. 
2. Install the requirements in requirements.txt
3. Install the quantitative investment platform [Qlib](https://github.com/microsoft/qlib) and download the data from Qlib:
	```
	# install Qlib from source
	pip install --upgrade  cython
	git clone https://github.com/microsoft/qlib.git && cd qlib
	python setup.py install
	
	# Download the stock features of Alpha360 from Qlib
	python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
	```
 ## Structure
1. `data`: 
  - `csi300_market_value_07to20.pkl`: Contains the market value data for the CSI 300 index from 2007 to 2020.
  - `csi300_stock2concept.npy`: Contains the mapping of CSI 300 stocks to their respective concepts.
  - `csi300_stock_index.npy`: Contains the mapping of CSI 300 stocks to their respective indexes.
2. `ALSGCN.py`: Contains the implementation of the ALSGCN model.
3. `dataloader.py`: Contains the data loading and preprocessing functions.
4. `learn_ALSGCN.py`: Contains the training and evaluation code for the ALSGCN model.
5. `requirements.txt`: Lists the required Python packages.
6. `utils.py`: Contains utility functions used in the project.

---

If you find this work useful in yours, please consider citing.

```
@inproceedings{yu2025alsgcn,
  title={{ALSGCN}: An Attention-based Long and Short-term Graph Convolutional Network for Stock Recommendation},
  author={Yu, Junpeng and Yao, Wenjie and Li, Zhihao and Gao, Lele and Xiao, Wenyun and Wang, Hongnian},
  booktitle={2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year={2025},
  month={October},
  address={Vienna, Austria},
  organization={IEEE},
}
```



