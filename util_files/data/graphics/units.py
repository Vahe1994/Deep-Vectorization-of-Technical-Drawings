import numbers
import re

from util_files import warnings


class GraphicUnits:
    def __init__(self, value, attributes={}):
        self.value = value
        self.units = ''
        self.attributes = attributes

    def __copy__(self):
        return self.__class__(self.value)

    copy = __copy__

    def __add__(self, other):
        if isinstance(other, GraphicUnits):
            return self.__class__(self.value + other.as_units(self.__class__).value)
        else:
            return self.__class__(self.value + other)

    def __radd__(self, other):
        return self.__class__(other + self.value)

    def as_units(self, cls):
        if cls == Pixels:
            return self.as_pixels()
        elif cls == Points:
            return self.as_points()

    def __lt__(self, other):
        if isinstance(other, GraphicUnits):
            return self.value < other.as_units(self.__class__).value
        else:
            return self.value < other

    def __le__(self, other):
        if isinstance(other, GraphicUnits):
            return self.value <= other.as_units(self.__class__).value
        else:
            return self.value <= other

    def __eq__(self, other):
        if isinstance(other, GraphicUnits):
            return self.value == other.as_units(self.__class__).value
        else:
            return self.value == other

    def __gt__(self, other):
        if isinstance(other, GraphicUnits):
            return self.value > other.as_units(self.__class__).value
        else:
            return self.value > other

    def __ge__(self, other):
        if isinstance(other, GraphicUnits):
            return self.value >= other.as_units(self.__class__).value
        else:
            return self.value >= other

    def __float__(self):
        return float(self.value)

    def __iadd__(self, other):
        if isinstance(other, GraphicUnits):
            self.value += other.as_units(self.__class__).value
        else:
            self.value += other
        return self

    def __imul__(self, other):
        self.value *= other
        return self

    def __int__(self):
        return int(self.value)

    def __invert__(self):
        return self.__class__(-self.value)

    def __mul__(self, other):
        return self.__class__(self.value * other)

    def __neg__(self):
        return self.__class__(-self.value)

    def __repr__(self):
        return str(self)

    __rmul__ = __mul__

    def __round__(self, ndigits=None):
        return int(round(self.value, ndigits))

    def __str__(self):
        return str(self.value) + self.units

    def __sub__(self, other):
        if isinstance(other, GraphicUnits):
            return self.__class__(self.value - other.as_units(self.__class__).value)
        else:
            return self.__class__(self.value - other)

    def __rsub__(self, other):
        return self.__class__(other - self.value)

    def __truediv__(self, other):
        if isinstance(other, GraphicUnits):
            return self.value / other.as_units(self.__class__).value
        else:
            return self.value / other

    def __rtruediv__(self, other):
        warnings.warn('Division of number by unit', warnings.UndefinedWarning)
        return other / self.value

    def scale(self, other):
        if ('constant' not in self.attributes) or (not self.attributes['constant']):
            self.value *= other
        return self


class Pixels(GraphicUnits):
    def __init__(self, value, attributes={}):
        super().__init__(value, attributes=attributes)
        self.units = 'px'

    def as_pixels(self):
        return self

    def as_points(self):
        return Points(self.value * 3 / 4, attributes=self.attributes)


class Points(GraphicUnits):
    def __init__(self, value, attributes={}):
        super().__init__(value, attributes=attributes)
        self.units = 'pt'

    def as_pixels(self):
        return Pixels(self.value * 4 / 3, attributes=self.attributes)

    def as_points(self):
        return self


_dim_repr = re.compile('^((?:[+-]?)(?:\\d*\\.\\d+|\\d+)(?:e[+-]?\\d+)?)(.*)$', re.I | re.U | re.X)


def fromrepr(representation, default_units=None):
    if isinstance(representation, numbers.Number):
        value = representation
        dim = ''
    else:
        value, dim = _dim_repr.findall(str(representation).lower())[0]
        value = float(value)

    if int(value) == value:
        value = int(value)

    if dim == '':
        if default_units is not None:
            return default_units(value)
        else:
            return GraphicUnits(value)
    elif dim == 'px':
        return Pixels(value)
    elif dim == 'pt':
        return Points(value)
    else:
        raise ValueError('Unknown dimension units "{}"'.format(dim))
