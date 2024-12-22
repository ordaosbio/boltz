#!/bin/bash

perl -i -p0e 's/requests\.post\(.*?(?=\))\)/requests\.post\(\"\#\"\,data\=None\)/s' src/boltz/data/msa/mmseqs2.py
