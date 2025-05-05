#!/usr/bin/env python
# Vidrovr Inc.

from multiprocessing import Pool

from tqdm import tqdm


def run_imap(func, argument_list, num_processes):
    """Calls a function against a list of arguments using imap and tqdm."""
    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(
        pool.imap(func=func, iterable=argument_list), total=len(argument_list)
    ):
        result_list_tqdm.append(result)

    return result_list_tqdm
