"""<dir>/<i>.inner.png + .alpha.png -> <out>/<i>.png colour share maps (orange = outer, blue = inner, white = empty)."""
import sys
from pathlib import Path
import cv2
import numpy as np
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ora, blue = np.array([240, 130, 20], np.float32), np.array([40, 90, 230], np.float32)
for p in sorted(src.glob("[0-9][0-9][0-9][0-9].inner.png")):
    sh = cv2.imread(str(p), 0)[..., None] / 255.0; al = cv2.imread(str(p).replace(".inner", ".alpha"), 0)[..., None] / 255.0
    img = al * ((1 - sh) * ora + sh * blue) + (1 - al) * 255
    cv2.imwrite(str(out / p.name.replace(".inner", "")), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
