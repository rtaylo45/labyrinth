#!/usr/bin/env python
# Vidrovr Inc.


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out")
