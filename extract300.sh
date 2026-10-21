#!/bin/sh

ls -1 plism-dataset/*P* | xargs -P13 -I% ./sample_tiles.py % 300-% 300 42
