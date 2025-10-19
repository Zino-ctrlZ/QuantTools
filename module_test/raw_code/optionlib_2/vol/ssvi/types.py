from enum import Enum
class VolSide(str, Enum):
    CALL = 'call'
    PUT = 'put'
    OTM = 'otm'

class VolType(str, Enum):
    BS = 'bs'
    BINOMIAL = 'binomial'

class DivType(str, Enum):
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'
