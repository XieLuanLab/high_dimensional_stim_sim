import itertools

def sort_and_groupby(iterable, key):
    # sorted_iterable = sorted(iterable, key=key)
    # return itertools.groupby(sorted_iterable, key=key)
    for (kname, elems) in itertools.groupby(sorted(iterable, key=key), key=key):
        yield kname, list(elems)
