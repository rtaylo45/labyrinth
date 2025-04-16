#!/usr/bin/env python
# Vidrovr Inc.

import numpy as np

rng = np.random.default_rng()


class RGBColorGenerator:
    def __init__(self, color_min: int = 0, color_max: int = 255):
        assert color_min >= 0 and color_min <= 255
        assert color_max >= 0 and color_max <= 255
        assert color_min < color_max

        if (color_min < 0) or (color_min > 255):
            raise ValueError(f"color_min ({color_min}) out of range [0, 255]")

        if (color_max < 0) or (color_max > 255):
            raise ValueError(f"color_max ({color_max}) out of range [0, 255]")

        if color_min >= color_max:
            raise ValueError(
                f"color_min ({color_min}) must be larger than color_max ({color_max})"
            )

        self.color_min = color_min
        self.color_max = color_max

    def __call__(
        self,
    ) -> tuple[int, int, int]:
        r = int(rng.integers(self.color_min, self.color_max))
        g = int(rng.integers(self.color_min, self.color_max))
        b = int(rng.integers(self.color_min, self.color_max))

        return (r, g, b)


class RGBAColorGenerator:
    constant_alpha: bool

    def __init__(
        self,
        color_min: int = 0,
        color_max: int = 255,
        alpha_min: int = 0,
        alpha_max: int = 255,
    ):
        assert color_min >= 0 and color_min <= 255
        assert color_max >= 0 and color_max <= 255
        assert color_min < color_max

        if (color_min < 0) or (color_min > 255):
            raise ValueError(f"color_min ({color_min}) out of range [0, 255]")

        if (color_max < 0) or (color_max > 255):
            raise ValueError(f"color_max ({color_max}) out of range [0, 255]")

        if (alpha_min < 0) or (alpha_min > 255):
            raise ValueError(f"alpha_min ({alpha_min}) out of range [0, 255]")

        if (alpha_max < 0) or (alpha_max > 255):
            raise ValueError(f"alpha_max ({alpha_max}) out of range [0, 255]")

        self.constant_alpha = False
        if alpha_min == alpha_max:
            self.constant_alpha = True

        self.color_min = color_min
        self.color_max = color_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def __call__(
        self,
    ) -> tuple[int, int, int, int]:
        r = int(rng.integers(self.color_min, self.color_max))
        g = int(rng.integers(self.color_min, self.color_max))
        b = int(rng.integers(self.color_min, self.color_max))

        if self.constant_alpha:
            a = int(self.alpha_min)
        else:
            a = int(rng.integers(self.alpha_min, self.alpha_max))

        return (r, g, b, a)
