"""Run `python -m ftgspp.train <cfg> <run>` with a WHITE background (in-memory patch, no external file modified;
docs/DECISIONS.md D-011). FTGS++ has no background option: training on black composites lets dark floaters sit
in free space at zero cost, which then show up under the white evaluation protocol.

The patch keeps FTGS's own rasterisation call untouched (packed mode is needed for its densification state) and
composites the returned premultiplied image onto white with the returned alpha: img + (1 - alpha) * 1, which is
exactly what a background colour does. Only RGB renders (sh_degree set) are affected; flow renders are untouched.
"""
import runpy
import sys

import gsplat

_orig = gsplat.rasterization


def _white(*args, **kw):
    img, alpha, meta = _orig(*args, **kw)
    if kw.get("sh_degree") is not None and kw.get("backgrounds") is None:
        img = img + (1.0 - alpha)
    return img, alpha, meta


gsplat.rasterization = _white
if __name__ == "__main__":
    sys.argv = ["ftgspp.train"] + sys.argv[1:]
    runpy.run_module("ftgspp.train", run_name="__main__", alter_sys=True)
