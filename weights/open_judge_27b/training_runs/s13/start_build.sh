#!/bin/bash
cd /home/ubuntu/vbrepo2/tools
setsid nohup /home/ubuntu/cc/bin/python -u cc_build.py --data /home/ubuntu/ccdata/data --out /home/ubuntu/cccases --want 1500 --workers 96 > /home/ubuntu/s13/ccbuild.log 2>&1 < /dev/null &
