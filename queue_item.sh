#!/bin/bash
echo "queueing" "${1}" "${2}" "${3}" "${4}"
/home/user/.pyenv/shims/python ~/src/pricing/producer.py "${1}" "${2}" "${3}" "${4}"
