# README

This is my interpretation of the [RCPCC](https://arxiv.org/abs/2502.06123), more specifically part B _Point Cloud Encoding_ in section III _Methodology_.

## Setup

This code runs on Python 3.11+.

```
1. Start new venv
$ python3 -m venv venv & source venv/bin/activate

2. Download required packages
$ pip install -r requirements.txt
```

## Execute

Run
```
$ python3 run.py --file-path <path-to-data>
```

## Notes

The data found in `/data` are extracted from the KITTI dataset. Any point cloud data that follows their format (`.bin`) should work with the script.