"""Extended Gymnasium spaces with state tracking and constraint support."""

import time

from gymnasium import spaces
from loguru import logger as logging

import stable_worldmodel as swm


class Discrete(spaces.Discrete):
    """Extended discrete space with state tracking and constraint support."""

    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        """Initialize a Discrete space with state tracking."""
        super().__init__(*args, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self):
        """The initial value of the space."""
        return self._init_value

    @property
    def value(self):
        """The current value of the space."""
        return self._value

    def reset(self):
        """Reset the space value to its initial value."""
        self._value = self.init_value

    def contains(self, x):
        """Check if value is valid and satisfies constraints."""
        return super().contains(x) and self.constrain_fn(x)

    def check(self):
        """Validate the current space value."""
        if not self.constrain_fn(self.value):
            logging.warning(f"Discrete: value {self.value} does not satisfy constrain_fn")
            return False
        return super().contains(self.value)

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        """Sample a random value using rejection sampling for constraints."""
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(*args, **kwargs)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")

    def set_init_value(self, value):
        """Set the initial value of the Discrete space."""
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the Discrete space")
        self._init_value = value

    def set_value(self, value):
        """Set the current value of the Discrete space."""
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the Discrete space")
        self._value = value


class MultiDiscrete(spaces.MultiDiscrete):
    """Extended multi-discrete space with state tracking and constraint support."""

    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        """Initialize a MultiDiscrete space with state tracking."""
        super().__init__(*args, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self):
        """The initial values of the space."""
        return self._init_value

    @property
    def value(self):
        """The current values of the space."""
        return self._value

    def reset(self):
        """Reset the space values to their initial values."""
        self._value = self.init_value

    def contains(self, x):
        """Check if values are valid and satisfy constraints."""
        return super().contains(x) and self.constrain_fn(x)

    def check(self):
        """Validate the current space values."""
        if not self.constrain_fn(self.value):
            logging.warning(f"MultiDiscrete: value {self.value} does not satisfy constrain_fn")
            return False
        return super().contains(self.value)

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        """Sample random values using rejection sampling for constraints."""
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(*args, **kwargs)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")

    def set_init_value(self, value):
        """Set the initial values of the MultiDiscrete space."""
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the MultiDiscrete space")
        self._init_value = value

    def set_value(self, value):
        """Set the current values of the MultiDiscrete space."""
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the MultiDiscrete space")
        self._value = value


class Box(spaces.Box):
    """Extended continuous box space with state tracking and constraint support."""

    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        """Initialize a Box space with state tracking."""
        super().__init__(*args, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._init_value = init_value
        self._value = init_value

    @property
    def init_value(self):
        """The initial value of the space."""
        return self._init_value

    @property
    def value(self):
        """The current value of the space."""
        return self._value

    def reset(self):
        """Reset the space value to its initial value."""
        self._value = self.init_value

    def contains(self, x):
        """Check if value is valid and satisfies constraints."""
        return super().contains(x) and self.constrain_fn(x)

    def check(self):
        """Validate the current space value."""
        if not self.constrain_fn(self.value):
            logging.warning(f"Box: value {self.value} does not satisfy constrain_fn")
            return False
        return self.contains(self.value)

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        """Sample a random value using rejection sampling for constraints."""
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(*args, **kwargs)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")

    def set_init_value(self, value):
        """Set the initial value of the Box space."""
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the Box space")
        self._init_value = value

    def set_value(self, value):
        """Set the current value of the Box space."""
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the Box space")
        self._value = value


class RGBBox(Box):
    """Specialized box space for RGB image data."""

    def __init__(self, shape=(3,), *args, init_value=None, **kwargs):
        if not any(dim == 3 for dim in shape):
            raise ValueError("shape must have a channel of size 3")

        super().__init__(
            low=0,
            high=255,
            shape=shape,
            dtype="uint8",
            init_value=init_value,
            *args,
            **kwargs,
        )


class Dict(spaces.Dict):
    """Extended dictionary space with ordered sampling and nested support."""

    def __init__(self, *args, init_value=None, constrain_fn=None, sampling_order=None, **kwargs):
        """Initialize a Dict space with state tracking and sampling order."""
        super().__init__(*args, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._init_value = init_value
        self._value = self.init_value

        # add missing keys
        if sampling_order is None:
            self._sampling_order = list(self.spaces.keys())
        elif len(sampling_order) != len(self.spaces):
            missing_keys = set(self.spaces.keys()).difference(set(sampling_order))
            logging.warning(
                f"Dict sampling_order is missing keys {missing_keys}, adding them at the end of the sampling order"
            )
            self._sampling_order = list(sampling_order) + list(missing_keys)
        else:
            self._sampling_order = sampling_order

        if not all(key in self.spaces for key in self._sampling_order):
            missing = set(self._sampling_order) - set(self.spaces.keys())
            raise ValueError(f"sampling_order contains keys not in spaces: {missing}")

    @property
    def init_value(self):
        """Initial values for all contained spaces."""
        init_val = {}

        for k, v in self.spaces.items():
            if hasattr(v, "init_value"):
                init_val[k] = v.init_value
            else:
                logging.warning(
                    f"Space {k} of type {type(v)} does not have init_value property, using default sample instead"
                )
                init_val[k] = v.sample()

        return init_val

    @property
    def value(self):
        """Current values of all contained spaces."""
        val = {}
        for k, v in self.spaces.items():
            if hasattr(v, "value"):
                val[k] = v.value
            else:
                raise ValueError(f"Space {k} of type {type(v)} does not have value property")
        return val

    def _get_sampling_order(self, parts=None):
        """Yield dotted paths for nested Dict space respecting sampling order."""
        if parts is None:
            parts = ()

        # Prefer an explicit sampling order; otherwise preserve insertion order.
        keys = getattr(self, "_sampling_order", None) or self.spaces.keys()

        for key in keys:
            # Skip if the key isn't in the mapping (defensive against stale order lists).
            if key not in self.spaces:
                continue

            key_str = str(key)  # ensure joinable
            path = parts + (key_str,)
            yield ".".join(path)

            subspace = self.spaces[key]
            if isinstance(subspace, spaces.Dict):
                # Recurse into nested Dict spaces
                yield from subspace._get_sampling_order(path)

    @property
    def sampling_order(self):
        """Set of dotted paths for all variables in sampling order."""
        return list(self._get_sampling_order())

    def reset(self):
        """Reset all contained spaces to their initial values."""
        for v in self.spaces.values():
            if hasattr(v, "reset"):
                v.reset()
        self._value = self.init_value

    def contains(self, x) -> bool:
        """Check if value is a valid member of this space."""
        if not isinstance(x, dict):
            return False

        for key in self.spaces.keys():
            if key not in x:
                return False

            if not self.spaces[key].contains(x[key]):
                return False

        if not self.constrain_fn(x):
            return False

        return True

    def check(self, debug=False):
        """Validate all contained spaces' current values."""
        for k, v in self.spaces.items():
            if hasattr(v, "check"):
                if not v.check():
                    if debug:
                        logging.warning(f"Dict: space {k} failed check()")
                    return False
        return True

    def names(self):
        """Return all space keys including nested ones."""

        def _key_generator(d, parent_key=""):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, spaces.Dict):
                    yield from _key_generator(v.spaces, new_key)
                else:
                    yield new_key

        return list(_key_generator(self.spaces))

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        """Sample a random element from the Dict space."""
        start = time.time()
        for i in range(max_tries):
            sample = {}

            for k in self._sampling_order:
                sample[k] = self.spaces[k].sample(*args, **kwargs, set_value=set_value)

            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample

            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling is taking a while...")

        raise RuntimeError(f"constrain_fn not satisfied after {max_tries} draws")

    def update(self, keys):
        """Update specific keys in the Dict space by resampling them."""

        keys = set(keys)
        order = self.sampling_order

        if len(keys) == 1 and "all" in keys:
            self.sample()
        else:
            for v in filter(keys.__contains__, order):
                try:
                    var_path = v.split(".")
                    swm.utils.get_in(self, var_path).sample()

                except (KeyError, TypeError):
                    raise ValueError(f"Key {v} not found in Dict space")

        assert self.check(debug=True), "Values must be within space!"

    def set_init_value(self, variations_values):
        """Set initial values for specific keys in the Dict space."""

        for k, v in variations_values.items():
            try:
                var_path = k.split(".")
                assert swm.utils.get_in(self, var_path).contains(v), (
                    f"Value {v} for key {k} is not contained in the space"
                )
                swm.utils.get_in(self, var_path).set_init_value(v)

            except (KeyError, TypeError):
                raise ValueError(f"Key {k} not found in Dict space")

    def set_value(self, variations_values):
        """Set current values for specific keys in the Dict space."""

        for k, v in variations_values.items():
            try:
                var_path = k.split(".")
                assert swm.utils.get_in(self, var_path).contains(v), (
                    f"Value {v} for key {k} is not contained in the space"
                )
                swm.utils.get_in(self, var_path).set_value(v)

            except (KeyError, TypeError):
                raise ValueError(f"Key {k} not found in Dict space")

    def to_str(self):
        def _tree(d, indent=0):
            lines = []
            for k, v in d.items():
                if isinstance(v, (dict | self.__class__ | spaces.Dict)):
                    lines.append("    " * indent + f"{k}:")
                    lines.append(_tree(v, indent + 1))
                else:
                    lines.append("    " * indent + f"{k}: {v}")
            return "\n".join(lines)

        return _tree(self.spaces)
