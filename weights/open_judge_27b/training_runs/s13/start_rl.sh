#!/bin/bash
cd /home/ubuntu/vbrepo2
setsid nohup /home/ubuntu/tr/bin/python -u tools/rl_cc.py --cases /home/ubuntu/cccases --work /home/ubuntu/s18 --iters 8 > /home/ubuntu/s13/rl_cc.out 2>&1 < /dev/null &
