#!/usr/bin/env bash
pip uninstall -y "seq2seq>=0.0.0"
python setup.py clean;
python setup.py install
python main.py
