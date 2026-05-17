import pytest

from labyrinth.backgrounds.color_generators import RGBASampler, RGBSampler


class TestRGBSampler:
    def test_returns_three_channel_tuple(self):
        color = RGBSampler()()
        assert len(color) == 3

    def test_values_within_default_range(self):
        sampler = RGBSampler()
        for _ in range(20):
            r, g, b = sampler()
            assert 0 <= r < 255
            assert 0 <= g < 255
            assert 0 <= b < 255

    def test_values_within_restricted_range(self):
        sampler = RGBSampler(color_min=50, color_max=100)
        for _ in range(20):
            r, g, b = sampler()
            assert 50 <= r < 100
            assert 50 <= g < 100
            assert 50 <= b < 100

    def test_returns_ints(self):
        r, g, b = RGBSampler()()
        assert isinstance(r, int)
        assert isinstance(g, int)
        assert isinstance(b, int)

    def test_seeded_is_reproducible(self):
        a = RGBSampler(seed=42)()
        b = RGBSampler(seed=42)()
        assert a == b

    def test_different_seeds_differ(self):
        a = RGBSampler(seed=1)()
        b = RGBSampler(seed=2)()
        assert a != b

    def test_color_min_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RGBSampler(color_min=-1)

    def test_color_max_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RGBSampler(color_max=256)

    def test_color_min_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            RGBSampler(color_min=100, color_max=50)

    def test_color_min_equal_to_max_raises(self):
        with pytest.raises(ValueError):
            RGBSampler(color_min=100, color_max=100)


class TestRGBASampler:
    def test_returns_four_channel_tuple(self):
        color = RGBASampler()()
        assert len(color) == 4

    def test_values_within_default_range(self):
        sampler = RGBASampler()
        for _ in range(20):
            r, g, b, a = sampler()
            assert 0 <= r < 255
            assert 0 <= g < 255
            assert 0 <= b < 255
            assert 0 <= a < 255

    def test_values_within_restricted_range(self):
        sampler = RGBASampler(color_min=50, color_max=100, alpha_min=200, alpha_max=250)
        for _ in range(20):
            r, g, b, a = sampler()
            assert 50 <= r < 100
            assert 50 <= g < 100
            assert 50 <= b < 100
            assert 200 <= a < 250

    def test_constant_alpha(self):
        sampler = RGBASampler(alpha_min=255, alpha_max=255)
        for _ in range(20):
            _, _, _, a = sampler()
            assert a == 255

    def test_returns_ints(self):
        r, g, b, a = RGBASampler()()
        assert isinstance(r, int)
        assert isinstance(g, int)
        assert isinstance(b, int)
        assert isinstance(a, int)

    def test_seeded_is_reproducible(self):
        a = RGBASampler(seed=42)()
        b = RGBASampler(seed=42)()
        assert a == b

    def test_different_seeds_differ(self):
        a = RGBASampler(seed=1)()
        b = RGBASampler(seed=2)()
        assert a != b

    def test_color_min_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RGBASampler(color_min=-1)

    def test_color_max_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RGBASampler(color_max=256)

    def test_alpha_min_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RGBASampler(alpha_min=-1)

    def test_alpha_max_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RGBASampler(alpha_max=256)

    def test_alpha_min_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            RGBASampler(alpha_min=200, alpha_max=100)
