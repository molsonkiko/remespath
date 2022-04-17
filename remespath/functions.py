from remespath.ast import *
from math import nan, inf
import math
import operator
import re
import typing


def negpow(x, y):
    '''pow has higher precedence than unary minus, unlike other binops'''
    return -(x**y)


def irange(start, stop, step):
    # maybe give this the option to take an iterable argument for convenient list(range(len(x)))?
    if stop is None:
        return list(range(start))
    if step is None:
        return list(range(start, stop))
    return list(range(start, stop, step))

def has_pattern(x, regex):
    return re.search(regex, x) is not None
    
def log(x, base):
    if base is None:
        return math.log(x)
    return math.log(x, base)

def max_by(x, key):
    return max(x, key = lambda x: x[key])
    
def min_by(x, key):
    return min(x, key = lambda x: x[key])
    
def sort_func(x, reverse):
    if reverse is None:
        return sorted(x)
    return sorted(x, reverse=reverse)
    
def sort_by(x, key, reverse):
    if reverse is None:
        return sorted(x, key=lambda x: x[key])
    return sorted(x, key=lambda x: x[key], reverse=reverse)
    
def flatten(x, iterations=None):
    if iterations is None or iterations == 1:
        assert isinstance(x, list), 'flatten only works on lists'
        out = []
        for e in x:
            if isinstance(e, list):
                out.extend(e)
            else:
                out.append(e)        
        return out
    out = x
    for ii in range(iterations):
        out = flatten(out)
    return out

def ifelse(condition, if_true, if_false):
    return if_true if condition else if_false

def str_len(x):
    assert isinstance(x, (str, bytes)), "strlen only takes string lengths"
    return len(x)

def str_mul(s, x):
    '''string multiplication, e.g., str_mul('ab', 3) = 'ababab'
    '''
    return s*x

def str_count(x, substr):
    if isinstance(substr, re.Pattern):
        return len(substr.findall(x))
    return str.count(x, substr)
    
def str_find(x, substr, how_many):
    if isinstance(substr, re.Pattern):
        return substr.findall(x)[slice(how_many)]
    return ([substr] * str.count(x, substr))[slice(how_many)]
    
def str_split(x, sep, maxsplit):
    maxsplit = maxsplit or -1
    if isinstance(sep, re.Pattern):
        return sep.split(x, maxsplit)
    return str.split(x, sep, maxsplit)

def str_sub(x, to_replace, repl, count):
    count = count or 0
    if isinstance(to_replace, re.Pattern):
        return to_replace.sub(repl, x)
    return str.replace(x, to_replace, repl)
    
def is_str(x):
    return isinstance(x, str)

def is_num(x):
    return isinstance(x, (int, bool, float))

def is_expr(x):
    return isinstance(x, (list, dict))


BINOPS = {
    # each value is a (function, precedence) pair.
    # thus (^, |, &) resolve after all other binops and pow resolves first
    # all of these are vectorized by default
    '&'  : (operator.and_, 0),
    '|'  : (operator.or_, 0),
    '^'  : (operator.xor, 0),
    'in' : (operator.contains, 1),
    '=~' : (has_pattern, 1),
    '==' : (operator.eq, 1),
    '!=' : (operator.ne, 1),
    '<'  : (operator.lt, 1),
    '>'  : (operator.gt, 1),
    '>=' : (operator.ge, 1),
    '<=' : (operator.le, 1),
    '+'  : (operator.add, 2),
    '-'  : (operator.sub, 2),
    '//' : (operator.floordiv, 3),
    '%'  : (operator.mod, 3),
    '*'  : (operator.mul, 3),
    '/'  : (operator.truediv, 3),
    '**' : (pow, 5),
    # '**-': (negpow, 5), 
}

BINOP_START_CHARS = set(x[0] for x in BINOPS)


CONSTANTS = {
    'NaN': nan,
    'false': False,
    'Infinity': inf,
    '-Infinity': -inf,
    'null': None,
    'true': True,
}

UMINUS_PRECEDENCE = 4
# every other unary function has higher precedence than all binops.
# but uminus has lower precedence than exponentiation, so we need to show that

class ArgFunction(typing.NamedTuple):
    func: typing.Callable
    out_type: str
    min_args: float
    max_args: float
    # if the number of args taken is inf, it's a *args function
    arg_types: typing.List[typing.Set[str]]
    is_vectorized: bool
    # if is_vectorized is True, func is applied to every element of an 
    # iterable separately


FUNCTIONS = {
    'flatten': ArgFunction(flatten, 'expr', 1, 2, [{'expr'}, {'int'}], False),
    'irange': ArgFunction(irange, 'expr', 1, 3, [{'int'}, {'int'}, {'int'}], False),
    'keys': ArgFunction(dict.keys, 'expr', 1, 1, [{'expr'}], False),
    'len': ArgFunction(len, 'int', 1, inf, [{'expr'}], False),
    'max': ArgFunction(max, 'num', 1, inf, [{'expr'}], False), 
    'max_by': ArgFunction(max_by, 'num', 2, 2, [{'expr'}, {'quoted_string', 'unquoted_string'}], False),
    'min': ArgFunction(min, 'num', 1, inf, [{'expr'}], False),
    'min_by': ArgFunction(min_by, 'num', 2, 2, [{'expr'}, {'quoted_string', 'unquoted_string'}], False),
    's_join': ArgFunction(str.join, 'quoted_string', 2, 2, [{'quoted_string'}, {'expr'}], False),
    'sort_by': ArgFunction(sort_by, 'expr', 2, 3, [{'expr'}, {'bool'}, {'quoted_string', 'unquoted_string'}], False),
    'sorted': ArgFunction(sort_func, 'expr', 1, 2, [{'expr'}, {'bool'}], False),
    'sum': ArgFunction(sum, 'num', 1, 1, [{'expr'}], False),
    'values': ArgFunction(dict.values, 'expr', 1, 1, [{'expr'}], False),
    # the below functions are all vectorized
    # the '?' for out_type means that the output type is same as input
    # if there's a list of two output types, the first is the output type if
    # it's an expr, the second is the output type if it's a scalar
    # '-': ArgFunction(operator.neg, '?', 1, 1, [{'expr', 'num', 'int'}], True),
    'abs': ArgFunction(abs, '?', 1, 1, [{'expr', 'num', 'int'}], True),
    'float': ArgFunction(float, ['expr', 'num'], 1, 1, [{'expr', 'int', 'quoted_string', 'num', 'bool'}], True),
    'ifelse': ArgFunction(ifelse, '?', 3, 3, [{'expr', 'bool'}, SCALAR_SUBTYPES, SCALAR_SUBTYPES], True),
    'int': ArgFunction(int, ['expr', 'int'], 1, 1, [{'expr', 'num', 'quoted_string', 'int', 'bool'}], True),
    'is_expr': ArgFunction(bool, ['expr', 'bool'], 1, 1, [ALL_BASE_SUBTYPES], True),
    'is_num': ArgFunction(bool, ['expr', 'bool'], 1, 1, [ALL_BASE_SUBTYPES], True),
    'is_str': ArgFunction(bool, ['expr', 'bool'], 1, 1, [ALL_BASE_SUBTYPES], True),
    'isna': ArgFunction(math.isnan, ['expr', 'bool'], 1, 1, [{'expr', 'num'}], True),
    'log': ArgFunction(math.log, ['expr', 'num'], 1, 2, [{'expr', 'num', 'int'}], True),
    'log2': ArgFunction(math.log2, ['expr', 'num'], 1, 1, [{'expr', 'num', 'int'}], True),
    'not': ArgFunction(operator.not_, ['expr', 'bool'], 1, 1, [SCALAR_SUBTYPES], True),
    'round': ArgFunction(round, ['expr', 'num'], 1, 2, [{'expr', 'num', 'int', 'bool'}, 'int'], True),
    's_count': ArgFunction(str_count, ['expr', 'int'], 2, 2, [{'expr', 'quoted_string', 'unquoted_string'}, {'regex', 'quoted_string'}], True),
    's_find': ArgFunction(str_find, 'expr', 2, 3, [{'expr', 'quoted_string', 'unquoted_string'}, {'regex', 'quoted_string'}, {'int'}], True),
    's_len': ArgFunction(str_len, ['expr', 'int'], 1, 1, [{'expr', 'quoted_string', 'unquoted_string'}], True),
    # s_mul is unnecessary in Python, but it should be part of the language
    # spec for convenience in other languages
    's_mul': ArgFunction(str_mul, ['expr', 'quoted_string'], 2, 2, [{'expr', 'quoted_string', 'unquoted_string'}, {'int'}], True),
    's_slice': ArgFunction(str.__getitem__, ['expr', 'quoted_string'], 2, 2, [{'expr', 'quoted_string', 'unquoted_string'}, {'slicer', 'int'}], True),
    's_split': ArgFunction(str_split, 'expr', 2, 3, [{'expr', 'quoted_string', 'unquoted_string'}, {'regex', 'quoted_string'}, {'int'}], True),
    's_sub': ArgFunction(str_sub, ['expr', 'quoted_string'], 3, 4, [{'expr', 'quoted_string', 'unquoted_string'}, {'regex', 'quoted_string'}, {'quoted_string'}, {'int'}], False),
    'str': ArgFunction(str, ['expr', 'quoted_string'], 1, 1, [ALL_BASE_SUBTYPES], True),
}