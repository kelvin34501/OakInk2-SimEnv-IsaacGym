import contextlib


@contextlib.contextmanager
def dummy_pbar_context():
    res = type("", (), {})

    def anyfunc(*args, **kwargs):
        pass

    res.update = anyfunc
    res.reset = anyfunc
    res.close = anyfunc
    res.clear = anyfunc
    res.set_description = anyfunc
    try:
        yield res
    finally:
        pass


def dummy_pbar():
    def anyfunc(*args, **kwargs):
        pass

    res = type(
        "",
        (),
        {
            "update": anyfunc,
            "reset": anyfunc,
            "close": anyfunc,
            "clear": anyfunc,
            "set_description": anyfunc,
        },
    )

    return res


fmt = "{desc:<20}{percentage:3.0f}%|{bar:60}{r_bar}"


def progress_bar(*, verbose=False, total=None, pos=0, desc=None, ncols=120):
    import tqdm

    res = (
        tqdm.tqdm(
            total=total,
            position=pos,
            bar_format=fmt,
            desc=desc,
            ncols=ncols,
        )
        if verbose
        else dummy_pbar()
    )
    return res


def wrap_pbar(iterator, *, verbose=False, total=None, pos=0, desc=None, ncols=120):
    if not verbose:
        return iterator

    import tqdm

    res = tqdm.tqdm(
        iterable=iterator,
        total=total,
        position=pos,
        bar_format=fmt,
        desc=desc,
        ncols=ncols,
    )
    return res
