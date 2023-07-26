### Environment Setup

1.  Install Python-RVO2 library
	```
	wget https://github.com/sybrenstuvel/Python-RVO2/archive/master.zip && unzip master && rm master.zip
	cd Python-RVO2-master && pip install Cython && pip install -r requirements.txt
	python setup.py build && python setup.py install
	```

2.  Install CrowdNav environment
	```
	pip install -e .
	```

3.  Download data
	```
	cd crowd_nav && mkdir data/demonstration/ -p && cd data/demonstration
	pip install gdown && gdown https://drive.google.com/uc?id=1D2guAxD_EgrKnJFMcLSBkf10SOagz0mr
	```

### Repo Structure
```
├── README.md
|
├── docs
│   ├── SETUP.md              <- You are here
|
├── crowd_nav
│   ├── imitate.py            <- Train script
│   ├── test.py               <- Test script
|
|   ├── data
|   	├── demonstration            <- Folder that contains demontrastion data
|   	├── output            <- Folder that contains training output
|
|── crowd_sim                 <- Folder that contains crowd nav simulation env
```
