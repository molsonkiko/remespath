'''
TODO:
1. Figure out formal ABNF description of query language
2. Finish building parser
    a. add support for projections
    b. make indexers work
    c. allow for indexing within expr functions
3. Add querying function

An attempt to combine the querying power of gorp.jsonpath with the intuitive
syntax of R dataframe indexing.

>>> from remespath import search
>>> j = {
... "foo": [
...     1,
...     False,
...         {"b": 0.5, "c": "a", "ba": 1.5, "co": {"de": True}},
...     None
...     ],
... "bar": ["baz", "quz"]
... }
>>> search("bar", j) # key search doesn't have to be in quotes.
['baz', 'quz']
>>> search('`bar`', j) # `` surround string literals
['baz', 'quz']
>>> search("foo[0]", j) # [] surround indices in an iterable
1
>>> search("foo[:1], j) # [] can also surround slices, following standard
... # Python slicing syntax.
>>> search("..b", j) # '..' turns on recursive search
0.5
>>> search('foo[2].g``', j) # "g" before a string literal makes it a regex
... # since the empty regex matches everything, this selects all keys.
[0.5, "a", 1.5, {"de": False}]
>>> search("foo[@ == 0]", j) # the [?<expr>] syntax applies filters
... # the @ inside the filter refers to the currently filtered object.
[False]
>>> search('bar[@ =~ `.a`]', j) # the "=~" operator is a regex-matching
... # operator for values
"baz"
>>> search('j`["baz", "buz"]`', j) # "j" before a string literal means
... # that it is parsed as JSON
['baz', 'buz']
>>> search('bar == j`["baz", "buz"]`', j) # comparison of two
... # equal-length vectors creates a boolean index
[True, False]
>>> search('bar[@ == j`["baz", "buz"]`', j) # using the boolean index
['baz']
>>> search('j`[1,2,3]` * 2', j) # arithmetic is vectorized across arrays
[2,4,6]
>>> search('`ab` * 2', j) # strings can be multiplied
'abab'
>>> search('j`{"a": 1, "b": 2}` * 2', j) # arithmetic is vectorized across dicts
{"a": 2, "b": 4}
>>> search('j`{"a": 1, "b": 2}` * j`{"a": 3, "b": 3}`', j) # you can apply any
... # binary function (e.g. addition, multiplication) to any two objects
... # with matching keys or any two arrays with equal lengths
... # obviously if type compatibility is violated an error will be raised
{"a": 3, "b": 6}
'''
from remespath.ast import *
from remespath.functions import *
from remespath.lexer import *
import operator
import re
import unittest
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s') #, filename="remespath.log", filemode='w')
RemesPathLogger = logging.getLogger(name='RemesPathLogger')


NUM_START_CHARS = set('-0123456789')
num_match = num_regex = re.compile(("(-?(?:0|[1-9]\d*))"   # any int or 0
                "(\.\d+)?" # optional decimal point and digits after
                "([eE][-+]?\d+)?" # optional scientific notation
                )).match
string_match = re.compile((
    r'((?:[^`]' # any number of (non-backtick characters...
    r'|(?<=\\)`)*)' # or backticks preceded by a backslash)
    r'(?<!\\)`')).match # terminated by a backtick not preceded by a backslash
int_match = re.compile('(-?\d+)').match

string_match = re.compile("[a-zA-Z_][a-zA-Z_0-9]*").match

EXPR_FUNC_ENDERS = {']', ':', '}', ',', ')'}
# these tokens have high enough precedence to stop an expr_function or scalar_function
INDEXER_STARTERS = {'.', '[', '{'}


def is_callable(x):
    return hasattr(x, '__call__')


class RemesParserException(Exception):
    def __init__(self, message, ind=None, x=None):
        self.x = x
        self.ind = ind
        self.message = message

    def __str__(self):
        if self.ind is None and self.x is None:
            return self.message
        return f'{self.message} (position {self.ind})\n{repr(self.x)}\n{self.x[self.ind]}' #{space_before_caret*" "}^'


class VectorizedArithmeticException(Exception):
    pass


##############
## INDEXER FUNCTIONS
##############

def apply_multi_index(inds):
    '''x: a list or dict
inds: a list of indices or keys to select'''
    def outfunc(x, inds):
        RemesPathLogger.debug(f"In apply_multi_index, x = {x}, inds = {inds}")
        if isinstance(x, dict):
            out = {}
            for k in inds:
                if isinstance(k, re.Pattern):
                    out.update(apply_regex_index(x, k))
                elif k in x:
                    out[k] = x[k]
            return out
        elif isinstance(x, list):
            out = []
            for k in inds:
                if isinstance(k, int):
                    if k < len(x):
                        out.append(x[k])
                elif isinstance(k, slice):
                    # k is a slice
                    out.extend(x[k])
            return out
        # x is a function
        elif is_callable(inds):
            return apply_multi_index(x, inds(x))
        raise RemesParserException("Expected x to be a list or dict, and for inds to be a list, dict or function")
    return lambda x: outfunc(x, inds)


def apply_boolean_index(inds):
    def outfunc(x, inds):
        if is_callable(inds):
            indfunc = inds
            inds = indfunc(x)
        if len(inds) != len(x):
            raise VectorizedArithmeticException(f"Boolean index length ({len(inds)}) != object/array length ({len(x)})")
        try:
            if isinstance(x, dict):
                out = {}
                for k, v in x.items():
                    ind = inds.get(k, False)
                    if not isinstance(ind, bool):
                        raise VectorizedArithmeticException('boolean index contains non-booleans')
                    if ind:
                        out[k] = v
            elif isinstance(x, list):
                out = []
                for v, ind in zip(x, inds):
                    if not isinstance(ind, bool):
                        raise VectorizedArithmeticException('boolean index contains non-booleans')
                    if ind:
                        out.append(v)
        except Exception as ex:
            raise VectorizedArithmeticException(str(ex))
        return out
    return lambda x: outfunc(x, inds)


def apply_regex_index(obj, regex):
    return {k: v for k, v in obj.items() if regex.search(k)}


def apply_indexer_list(indexers):
    '''recursively search an object to match indexers'''
    def outfunc(obj, indexers):
        idxr, k, has_one_option, is_projection = indexers[0]
        result = idxr(obj)
        if is_projection:
            if len(indexers) == 1:
                return result
            return outfunc(result, indexers[1:])
        RemesPathLogger.info(f"In apply_indexer_list.outfunc, idxr = {idxr}, obj = {obj}, result = {result}")
        is_dict = isinstance(result, dict)
        if len(indexers) == 1:
            if result and has_one_option:
                if not is_callable(result):
                    return result[k]
                return result(obj)[k]
            if not is_callable(result):
                return result
            return result(obj)
        if is_dict:
            if has_one_option:
                # don't need to specify the key when only one key is possible
                return outfunc(result[k], indexers[1:])
            out = {}
            for k, v in result.items():
                subdex = outfunc(v, indexers[1:])
                if subdex:
                    out[k] = subdex
            return out
        # obj is a list
        if has_one_option:
            # don't need an array output when you're getting a single index
            return outfunc(result[0], indexers[1:])
        out = []
        for v in result:
            subdex = outfunc(v, indexers[1:])
            if subdex:
                out.append(subdex)
        return out
        RemesPathLogger.info(f"apply_indexer_list.outfunc returns {result}")
    return lambda obj: outfunc(obj, indexers)


###############
## VECTORIZED STUFF
###############

def binop_two_jsons(func, a, b):
    if not is_expr(a):
        if not is_expr(b):
            return func(a, b)
        return binop_scalar_json(func, a, b)
    elif not is_expr(b):
        return binop_json_scalar(func, a, b)
    la, lb = len(a), len(b)
    if la !=  lb :
        raise VectorizedArithmeticException(f"Tried to apply a binop to two objects of lengths {la} and {lb}")
    if isinstance(a, dict):
        try:
            return {k: func(a[k], b[k]) for k in a}
        except KeyError as ex:
            raise VectorizedArithmeticException(str(ex))
    return [func(a[ii], b[ii]) for ii in range(la)]


def binop_scalar_json(func, s, j):
    if isinstance(j, dict):
        return {k: func(s, v) for k, v in j.items()}
    return [func(s, v) for v in j]


def binop_json_scalar(func, j, s):
    if isinstance(j, dict):
        return {k: func(v, s) for k, v in j.items()}
    return [func(v, s) for v in j]


def resolve_binop(binop, a, b): # , obj):
    '''apply a binop to two args that are exprs or scalars'''
    RemesPathLogger.info(f"In resolve_binop, binop = {binop}, a = {a}, b = {b}")
    aval, bval = a['value'], b['value']
    atype, btype = a['type'], b['type']
    out = None
    if atype in SCALAR_SUBTYPES:
        if btype in SCALAR_SUBTYPES:
            outval = binop(aval, bval)
            out = AST_TYPE_BUILDER_MAP[type(outval)](outval)
        elif btype == 'cur_json':
            out = cur_json(lambda obj: binop_scalar_json(binop, aval, bval(obj)))
        else:
            out = expr(binop_scalar_json(binop, aval, bval))
    elif atype == 'cur_json':
        if btype in SCALAR_SUBTYPES:
            out = cur_json(lambda obj: binop_json_scalar(binop, aval(obj), bval))
        elif btype == 'cur_json':
            out = cur_json(lambda obj: binop_two_jsons(binop, aval(obj), bval(obj)))
        else:
            out = cur_json(lambda obj: binop_two_jsons(binop, aval(obj), bval))
    else:
        if btype in SCALAR_SUBTYPES:
            out = expr(binop_json_scalar(binop, aval, bval))
        elif btype == 'cur_json':
            out = cur_json(lambda obj: binop_two_jsons(binop, aval, bval(obj)))
        else:
            out = expr(binop_two_jsons(binop, aval, bval))
    RemesPathLogger.info(f"resolve_binop returns {out}")
    return out


def resolve_binop_tree(binop, a, b):
    '''applies a binop to two args that are exprs, scalars, or other binops'''
    RemesPathLogger.info(f"In resolve_binop_tree, binop = {binop}, a = {a}, b = {b}")
    if a['type'] == 'binop':
        a = resolve_binop_tree(a['value'][0], *a['children']) # , obj)
    if b['type'] == 'binop':
        b = resolve_binop_tree(b['value'][0], *b['children']) # , obj)
    return resolve_binop(binop, a, b) # , obj)


def apply_arg_function(func, out_types, is_vectorized, *args): # inp, *args):
    RemesPathLogger.debug(f"In apply_arg_function, func = {func}, out_types={out_types}, is_vectorized={is_vectorized}, args={args}")
    if is_vectorized:
        avals = [a if not a else a['value'] for a in args]
        x = args[0]
        xval = avals[0]
        xtype = x['type']
        x_callable = is_callable(xval) # is a function
        other_callables = False
        other_args = []
        for arg in avals[1:]:
            if is_callable(arg):
                other_callables = True
            other_args.append(arg)
        if x_callable:
            out_types = 'cur_json'
        elif isinstance(out_types, list):
            out_types = out_types[0] if xtype == 'expr' else out_types[1]
        elif out_types == '?':
            out_types = xtype
        else:
            out_types = out_types
        ast_tok_builder = AST_TOK_BUILDER_MAP[out_types]
        if xtype == 'expr':
            if other_callables:
                if isinstance(xval, dict):
                    return ast_tok_builder(lambda inp: {k: func(v, *[a if not is_callable(a) else a(inp) for a in other_args]) for k, v in xval.items()})
                elif isinstance(xval, list):
                    return ast_tok_builder(lambda inp: [func(v, *[a if not is_callable(a) else a(inp) for a in other_args]) for v in xval])
            if isinstance(xval, dict):
                return ast_tok_builder({k: func(v, *other_args) for k, v in xval.items()})
            else:
                return ast_tok_builder([func(v, *other_args) for v in xval])
        elif x_callable:
            if other_callables:
                def outfunc(inp):
                    itbl = xval(inp)
                    othargs = [a if not is_callable(a) else a(inp) for a in other_args]
                    if isinstance(itbl, dict):
                        return {k: func(v, *othargs) for k, v in itbl.items()}
                    elif isinstance(itbl, list):
                        return [func(v, *othargs) for v in itbl]
                    return func(itbl, *othargs)
            else:
                def outfunc(inp):
                    itbl = xval(inp)
                    if isinstance(itbl, dict):
                        return {k: func(v, *other_args) for k, v in itbl.items()}
                    elif isinstance(itbl, list):
                        return [func(v, *other_args) for v in itbl]
                    return func(itbl, *other_args)
            return ast_tok_builder(outfunc)
        elif other_callables:
            return ast_tok_builder(lambda inp: func(xval, *[a if not is_callable(a) else a(inp) for a in other_args]))
        return ast_tok_builder(func(xval, *other_args))
    # the following is if it's NOT vectorized
    avals = [a if not a else a['value'] for a in args]
    x = args[0]
    other_args = []
    xval = avals[0]
    xtype = x['type']
    x_callable = is_callable(xval) # is a function
    other_callables = False
    for arg in avals[1:]:
        if is_callable(arg):
            other_callables = True
        other_args.append(arg)
    ast_tok_builder = AST_TOK_BUILDER_MAP[out_types]
    if x_callable:
        if other_callables:
            return ast_tok_builder(lambda inp: func(x(inp), *[a if not is_callable(a) else a(inp) for a in other_args]))
        return ast_tok_builder(lambda inp: func(xval(inp), *other_args))
    elif other_callables:
        return ast_tok_builder(lambda inp: func(xval, *[a if not is_callable(a) else a(inp) for a in other_args]))
    return ast_tok_builder(func(*avals))


################
## PARSER FUNCTIONS
################

def peek_next_token(x, ii):
    if ii + 1 >= len(x):
        return None
    return x[ii + 1]


def parse_slicer(query, ii, first_num):
    grps = []
    last_num = first_num
    end = ii
    while end < len(query):
        RemesPathLogger.debug(f'in parse_slicer, query[end] = {query[end]}, grps = {grps}')
        t = query[end]
        typ = t['type']
        if typ == 'delim':
            tval = t.get('value')
            if tval == ':':
                grps.append(last_num)
                last_num = None
                end += 1
                continue
            elif tval in EXPR_FUNC_ENDERS:
                break
        try:
            numtok, end = parse_expr_or_scalar_func(query, end)
            assert numtok['type'] == 'int'
            last_num = numtok['value']
        except:
            raise RemesParserException("Found non-integer while parsing slicer", ii, query)
        if len(grps) == 2:
            break
    grps.append(last_num)
    RemesPathLogger.debug(f'at parse_slicer return, query[end] = {query[end]}, end = {end}, grps = {grps}')
    return slicer(grps), end - 1


def parse_indexer(query, ii):
    t = query[ii]
    typ = t['type']
    tv = t.get('value')
    if tv == '.':
        nt = peek_next_token(query, ii)
        if nt and nt['type'] not in VARNAME_SUBTYPES:
            raise RemesParserException("'.' syntax for indexers only allows a single key or regex as indexer", ii, query)
        return varname_list([nt['value']]), ii + 2
    elif tv == '{':
        return parse_projection(query, ii + 1) 
    elif tv != '[':
        raise RemesParserException('Indexer must start with "." or "["', ii, query)
    children = []
    indexer = None
    last_tok = None
    last_type = None
    ii += 1
    while ii < len(query):
        t = query[ii]
        typ, tv = t['type'], t.get('value')
        RemesPathLogger.debug(f'in parse_indexer, last_tok = {last_tok}, ii = {ii}')
        if tv == ']':
            if not last_tok:
                raise RemesParserException("Empty indexer", ii, query)
            children.append(last_tok['value'])
            if not indexer:
                if last_type in VARNAME_SUBTYPES:
                    indexer = varname_list(children)
                elif last_type in ['slicer', 'int']:
                    indexer = slicer_list(children)
                else:
                    val = last_tok['value']
                    if is_callable(val):
                        indexer = cur_json(last_tok['value'])
                    else:
                        indexer = expr(last_tok['value'])
            if (indexer['type'] == 'slicer_list' and last_type not in SLICER_SUBTYPES) \
            or (indexer['type'] == 'varname_list' and last_type not in VARNAME_SUBTYPES):
                raise RemesParserException("Cannot have indexers with a mix of ints/slicers and strings", ii, query)
            return indexer, ii + 1
        elif last_tok and tv != ',' and tv != ':':
            raise RemesParserException("Consecutive indexers must be separated by commas", ii, query)
        elif tv == ',':
            # figure out how to deal with slices
            # if not all(x is None
            if not indexer:
                if last_type in VARNAME_SUBTYPES:
                    indexer = varname_list([])
                elif last_type in SLICER_SUBTYPES:
                    indexer = slicer_list([])
                children = indexer['children']
            elif (indexer['type'] == 'slicer_list' and last_type not in SLICER_SUBTYPES) \
            or (indexer['type'] == 'varname_list' and last_type not in VARNAME_SUBTYPES):
                raise RemesParserException("Cannot have indexers with a mix of ints/slicers and strings", ii, query)
            if not last_tok:
                raise RemesParserException("Comma before first entry in indexer list.", ii, query)
            children.append(last_tok['value'])
            last_tok = None
            last_type = None
            ii += 1
        elif tv == ':':
            if last_tok is None:
                last_tok, ii = parse_slicer(query, ii, None)
            elif last_type == 'int':
                last_tok, ii = parse_slicer(query, ii, last_tok['value'])
            else:
                raise RemesParserException(f"Expected token other than ':' after {last_tok} in an indexer", ii, query)
            ii += 1
            last_type = last_tok['type']
        else:
            last_tok, ii = parse_expr_or_scalar_func(query, ii)
            last_type = last_tok['type']
    raise RemesParserException("Unterminated indexer (EOF)")


def parse_expr_or_scalar(query, ii):
    '''Returns any single scalar or expr, including arg functions.
Does not resolve binops.'''
    if not query:
        raise RemesParserException("Empty query")
    t = query[ii]
    typ = t['type']
    tv = t.get('value')
    RemesPathLogger.info(f"in parse_expr_or_scalar, t = {t}, ii = {ii}")
    if typ == 'binop':
        raise RemesParserException("Binop without appropriate operands", ii, query)
    elif tv == '(':
        unclosed_parens = 1
        subquery = []
        for end in range(ii + 1, len(query)):
            subtok = query[end]
            subval = subtok.get('value')
            if subval == '(':
                unclosed_parens += 1
            elif subval == ')':
                unclosed_parens -= 1
                if unclosed_parens == 0:
                    last_tok, subii = parse_expr_or_scalar_func(subquery, 0)
                    ii = end + 1
                    break
            subquery.append(subtok)
        if unclosed_parens:
            raise RemesParserException("Unmatched '('", ii, query)
    elif typ == 'arg_function':
        last_tok, ii = parse_arg_function(query, ii+1, t)
    # elif typ == 'cur_json':
        # last_tok = expr(jsnode)
        # ii += 1
    else:
        last_tok = t
        ii += 1
    if last_tok['type'] in EXPR_SUBTYPES:
        idxrs = []
        cur_idxr = None
        # expr (indexer_list | projection)*
        nt = peek_next_token(query, ii - 1)
        # check if the expr has any indexers
        while nt and nt['type'] == 'delim' and nt['value'] in INDEXER_STARTERS:
            cur_idxr, ii = parse_indexer(query, ii)
            ixtype = cur_idxr['type']
            ixtype_end = ixtype[-4:]
            children = cur_idxr.get('children') or cur_idxr['value']
            has_one_option = False
            k = None
            is_projection = False
            if isinstance(children, list) and len(children) == 1 and isinstance(children[0], (int, str)):
                # it's more user-friendly to only return the value in a single
                # key-value pair or a single index from an array
                has_one_option = True
                k = children[0] if isinstance(children[0], str) else 0
            if ixtype_end == 'list':
                # varname_list (e.g. [bar,baz], .foo) or slicer_list (e.g. [0], [1,2:])
                idx_func = apply_multi_index(children)
            elif ixtype == 'expr':
                # a static boolean index (based on some JSON defined within the query)
                # something like j`[true,false,true]`
                idx_func = apply_boolean_index(children)
            elif ixtype_end == 'tion':
                is_projection = True
                if is_callable(children):
                    idx_func = children
                else:
                    idx_func = lambda obj: children
            else:
                # it's a boolean index based on the current object (a cur_json)
                # something like [@.bar <= @.baz]
                idx_func = apply_boolean_index(children)
            RemesPathLogger.info(f"In parse_expr_or_scalar, found indexer {cur_idxr}, k = {k}, children = {children}")
            idxrs.append((idx_func, k, has_one_option, is_projection))
            nt = peek_next_token(query, ii - 1)
        if idxrs:
            if last_tok['type'] == 'cur_json':
                result = lambda obj: apply_indexer_list(idxrs)(obj)
            else:
                result = apply_indexer_list(idxrs)(last_tok['value'])
            last_tok = AST_TYPE_BUILDER_MAP[type(result)](result)
    RemesPathLogger.info(f"parse_expr_or_scalar returns {(last_tok, ii)}")
    return last_tok, ii


def parse_expr_or_scalar_func(query, ii):
    '''handles scalars and exprs with binops'''
    uminus = False
    left_operand = None
    left_tok = None
    branch_children = None
    left_precedence = -inf
    root = None
    curtok = None
    children = None
    func = None
    precedence = None
    while ii < len(query):
        left_tok = curtok
        left_type = '' if not left_tok else left_tok['type']
        curtok = query[ii]
        tv = curtok.get('value')
        typ = curtok['type']
        RemesPathLogger.debug(f"In parse_expr_or_scalar_func, curtok = {curtok}")
        if typ == 'delim' and tv in EXPR_FUNC_ENDERS:
            if not left_tok:
                raise RemesParserException("No expression found where expr or scalar expected", ii, query)
            curtok = left_tok
            break
        if typ == 'binop':
            if left_tok is None or left_tok['type'] == 'binop':
                if tv[0] != operator.sub:
                    raise RemesParserException("Binop with invalid left operand", ii, query)
                uminus = not uminus
            else:
                children = curtok['children']
                func, precedence = tv
                show_precedence = precedence
                if func == pow:
                    show_precedence += 0.1
                    # to account for right associativity of exponentiation
                    if uminus:
                        # to account for exponentiation binding more tightly
                        # than unary '-'
                        curtok['value'][0] = negpow
                        uminus = False
                if left_precedence >= show_precedence:
                    # the left binop wins, so it takes the last operand.
                    # this binop becomes the root, and the next binop competes
                    # with it.
                    branch_children[1] = left_operand
                    children[0] = root
                    root = curtok
                else:
                    # the current binop wins, so it takes the left operand.
                    # the root stays the same, and the next binop competes
                    # with the curent binop
                    if not root:
                        root = curtok
                    else:
                        branch_children[1] = curtok
                    children[0] = left_operand
                left_precedence = precedence
                branch_children = children
            ii += 1
        else:
            if left_tok and left_tok['type'] != 'binop':
                raise RemesParserException("Can't have two exprs or scalars unseparated by a binop", ii, query)
            left_operand, ii = parse_expr_or_scalar(query, ii)
            if uminus:
                nt = peek_next_token(query, ii-1)
                if not (nt and nt['type'] == 'binop' and nt['value'][0] == pow):
                    # apply uminus to the left operand unless pow is coming
                    # up. uminus has higher precedence than other binops
                    left_operand = apply_arg_function(operator.neg, '?', True, left_operand)
                    uminus = False
            curtok = left_operand
        RemesPathLogger.info(f'In parse_expr_or_scalar_func, root = {root},curtok = {curtok}, uminus = {uminus}')
    if root:
        branch_children[1] = curtok
        RemesPathLogger.debug(f'parse_expr_or_scalar_func resolves binop:\n{root}')
        left_operand = resolve_binop_tree(root['value'][0], *root['children'])
    RemesPathLogger.debug(f'parse_expr_or_scalar_func returns {left_operand}')
    return left_operand, ii


def parse_arg_function(query, ii, argfunc):
    '''Handles functions that accept arguments using the "(" arg ("," arg)* ")"
syntax.
This is relevant to both the expr_function and scalar_function parts of the
grammar, because it is agnostic about the types of arguments received
    '''
    args = argfunc['children']
    func, out_types, min_args, max_args, arg_types, is_vectorized = argfunc['value']
    t = query[ii]
    if not (t['type'] == 'delim' and t['value'] == '('):
        raise RemesParserException(f"Function '{func.__name__}' must have parens surrounding arguments", ii, query)
    ii += 1
    arg_num = 0
    cur_arg = None
    while ii < len(query):
        t = query[ii]
        tv = t.get('value')
        type_options = arg_types[arg_num]
        # try to parse the current argument as one of the valid types
        RemesPathLogger.debug(f'In parse_arg_function, t = {t}, func = {func.__name__}, query[ii] = {query[ii]}, ii = {ii}, arg_num = {arg_num}')
        try:
            try:
                cur_arg, ii = parse_expr_or_scalar_func(query, ii)
            except:
                cur_arg = None
            if 'slicer' in type_options and tv == ':':
                cur_arg, ii = parse_slicer(query, ii, cur_arg)
                ii += 1
            if cur_arg is None or cur_arg['type'] not in type_options:
                argtype = None if cur_arg is None else cur_arg['type']
                raise RemesParserException(f"For arg {arg_num} of function {func.__name__}, expected argument of a type in {type_options}, instead got type {argtype}")
        except Exception as ex:
            raise RemesParserException(f"For arg {arg_num} of function {func.__name__}, expected argument of a type in {type_options}, instead raised exception:\n{str(ex)}")
        t = query[ii]
        tv = t.get('value')
        if arg_num + 1 < min_args and tv != ',':
            raise RemesParserException(f"Expected ',' after argument {arg_num} of function {func.__name__} ({min_args}-{max_args} args)", ii, query)
        if arg_num + 1 == max_args and tv != ')':
            raise RemesParserException(f"Expected ')' after argument {arg_num} of function {func.__name__} ({min_args}-{max_args} args)", ii, query)
        elif max_args < inf:
            args[arg_num] = cur_arg
        else:
            args.append(cur_arg)
        arg_num += 1
        ii += 1
        if tv == ')':
            RemesPathLogger.debug(f'at return of parse_arg_function, func = {func.__name__}, out_types = {out_types}, args = {args}')
            return apply_arg_function(func, out_types, is_vectorized, *args), ii
    raise RemesParserException(f"Expected ')' after argument {arg_num} of function {func.__name__} ({min_args}-{max_args} args)", ii, query)


def parse_projection(query, ii):
    children = []
    is_object_projection = False
    nt = None
    key, val = None, None
    tv, typ = None, None
    entry = None
    while ii < len(query):
        key, ii = parse_expr_or_scalar_func(query, ii)
        typ = key['type']
        tv = key.get('value')
        nt = peek_next_token(query, ii - 1)
        RemesPathLogger.debug(f"In parse_projection, nt = {nt}, key = {key}, ii = {ii}")
        is_delim = nt and nt['type'] == 'delim'
        if is_delim:
            if nt['value'] == ':':
                if children and not is_object_projection:
                    raise RemesParserException("Mixture of values and key-value pairs in an object/array projection", ii, query)
                if typ == 'string':
                    val, ii = parse_expr_or_scalar_func(query, ii + 1)
                    if not children:
                        children = {}
                    children[key['value']] = val['value']
                    is_object_projection = True
                    nt = peek_next_token(query, ii - 1)
                    RemesPathLogger.debug(f"In parse_projection, nt = {nt}, val = {val}, ii = {ii}")
                else:
                    raise RemesParserException(f"Object projection keys must be string, not {nt['type']}", ii, query)
            else:
                children.append(key['value'])
            if nt['value'] == '}':
                RemesPathLogger.debug(f"At return of parse_projection, children = {children}")
                if isinstance(children, dict):
                    def outfunc(obj):
                        out = {}
                        for k, v in children.items():
                            if is_callable(v):
                                out[k] = v(obj)
                            else:
                                out[k] = v
                        return out
                else:
                    outfunc = lambda obj: [v if not is_callable(v) else v(obj) for v in children]
                return projection(outfunc), ii + 1
            if nt['value'] != ',':
                raise RemesParserException("Values or key-value pairs in a projection must be comma-separated", ii, query)
        else:
            raise RemesParserException("Values or key-value pairs in a projection must be comma-separated", ii, query)
        ii += 1
    raise RemesParserException("Unterminated projection", len(query)-1, query)


def search(query, obj):
    toks = tokenize(query)
    result = parse_expr_or_scalar_func(toks, 0)
    resval = result[0]['value']
    if is_callable(resval):
        return resval(obj)
    return resval


#################
## TESTING
#################

def test_parse_indexer(tester, x, out):
    idx = parse_indexer(tokenize(x), 0)[0]
    if idx['type'] in EXPR_SUBTYPES:
        tester.assertEqual(out, idx['value'])
    else:
        tester.assertEqual(out, idx['children'])
        

def test_parse_arg_function(tester, funcname, argstring, obj, correct_out):
    toks = tokenize(argstring)
    func = arg_function(FUNCTIONS[funcname])
    result = parse_arg_function(toks, 0, func)
    resval = result[0]['value']
    if is_callable(resval):
        # output token value is a function, so call it
        tester.assertEqual(resval(obj), correct_out)
        return
    tester.assertEqual(result, (correct_out, len(toks)))
    
    
def test_parse_expr_or_scalar_func(tester, inp, obj, correct_out):
    toks = tokenize(inp)
    result = parse_expr_or_scalar_func(toks, 0)
    resval = result[0]['value']
    if is_callable(resval):
        # output token value is a function, so call it 
        tester.assertEqual(resval(obj), correct_out)
        return
    tester.assertEqual(result, (correct_out, len(toks)))
    

def test_apply_indexer_list(tester, idx_strings, obj, correct_out):
    idxrs = [parse_indexer(tokenize(ix), 0)[0] for ix in idx_strings]
    ix_k_hasopt = []
    for idxr in idxrs:
        children = idxr['children']
        has_one_option, k = False, None
        if len(idxr) == 1 and isinstance(children[0], (str, int)):
            has_one_option = True
            k = idxr[0] if isinstance(children[0], str) else 0
        ix_k_hasopt.append((apply_multi_index(children), k, has_one_option, False))
    tester.assertEqual(apply_indexer_list(ix_k_hasopt)(obj), correct_out)
    

class RemesPathTester(unittest.TestCase):
    ##############
    ## misc tests
    ##############
    def test_current_node(self):
        self.assertEqual(parse_expr_or_scalar(tokenize('@'), 0)[0]['value']([1]), [1])

    ##############
    ## indexer tests
    ##############
    def test_parse_indexer_one_string(self):
        test_parse_indexer(self, '.foo', ['foo'])

    def test_parse_indexer_dot_int_raises(self):
        with self.assertRaises(RemesParserException):
            parse_indexer(tokenize('.1'), 0)

    def test_parse_indexer_dot_slicer_raises(self):
        with self.assertRaises(RemesParserException):
            parse_indexer(tokenize('.:1'), 0)

    def test_parse_indexer_dot_opensqbk_raises(self):
        with self.assertRaises(RemesParserException):
            parse_indexer(tokenize('.[1]'), 0)

    def test_parse_indexer_one_string_endsqbk(self):
        test_parse_indexer(self, '[foo]', ['foo'])

    def test_parse_indexer_one_int(self):
        test_parse_indexer(self,  '[1]', [1])

    def test_parse_indexer_multi_int(self):
        test_parse_indexer(self, '[1,2,3]', [1,2,3])

    def test_parse_indexer_regex(self):
        test_parse_indexer(self, '.g`ab`', [re.compile('ab')])

    def test_parse_indexer_one_string_with_digits(self):
        test_parse_indexer(self, '.a2', ['a2'])

    def test_parse_indexer_multi_string_with_digits(self):
        test_parse_indexer(self, '[`a`,_3z2]', ['a', '_3z2'])

    def test_parse_indexer_json(self):
        test_parse_indexer(self, '[j`{"ab": 1}`]', {'ab': 1})

    def test_parse_indexer_multi_string(self):
        test_parse_indexer(self, '[foo,ba_r]', ['foo', 'ba_r'])

    def test_parse_indexer_regex_string(self):
        test_parse_indexer(self, '[g``,`bar`]', [re.compile(''), 'bar'])

    def test_parse_indexer_multi_backtick_string(self):
        test_parse_indexer(self, '[`ab`,`adf`,`fjf`]', ['ab', 'adf', 'fjf'])

    def test_parse_indexer_backtick_nobacktickstring(self):
        test_parse_indexer(self, '[`ab`,foo]', ['ab', 'foo'])

    def test_parse_indexer_escaped_backtick_edges(self):
        test_parse_indexer(self, '.`\\``', ['`'])

    def test_parse_indexer_escaped_backtick(self):
        test_parse_indexer(self, '.`a\\`b`', ['a`b'])

    def test_parse_indexer_escaped_backtick_regex(self):
        test_parse_indexer(self, '.g`a\\`b`', [re.compile('a`b')])

    def test_parse_indexer_escaped_backtick_json(self):
        test_parse_indexer(self, '[j`["\\`", 1]`]', ['`', 1])

    def test_parse_indexer_slicers(self):
        for sli, out in [
            (':', [slice(None,None,None)]),
            (':1', [slice(None,1,None)]),
            ('1', [1]),
            ('1:', [slice(1,None,None)]),
            ('0:2', [slice(0,2,None)]),
            ('-1:', [slice(-1,None,None)]),
            ('::2', [slice(None,None,2)]),
            (':-1', [slice(None,-1,None)]),
            (':1:-1', [slice(None,1,-1)]),
            ('1::-1', [slice(1,None,-1)]),
            ('1:2:1', [slice(1,2,1)]),
            ('::-1', [slice(None,None,-1)]),
            ('-3:5', [slice(-3, 5, None)]),
        ]:
            with self.subTest(sli=sli):
                test_parse_indexer(self, '[' + sli + ']', out)

    def test_parse_indexer_multi_slicers(self):
        test_parse_indexer(self, '[:1,2:]', [slice(None, 1), slice(2, None)])

    def test_parse_indexer_int_slicer(self):
        test_parse_indexer(self, '[1,2:]', [1, slice(2, None)])
        
    def test_parse_indexer_slicer_int(self):
        test_parse_indexer(self, '[2:,1]', [slice(2, None), 1])

    def test_parse_indexer_raises_int_string(self):
        # indexers containing a mix of strings and ints are disallowed
        for s in ['`a`', 'g`a`', 'a']:
            with self.subTest(s=s):
                with self.assertRaises(RemesParserException):
                    parse_indexer(tokenize(f'[2,{s}]'), 0)

    def test_parse_indexer_raises_slicer_string(self):
        for s in ['`a`', 'g`a`', 'a']:
            with self.subTest(s=s):
                with self.assertRaises(RemesParserException):
                    parse_indexer(tokenize(f'[2:,{s}]'), 0)

    def test_parse_indexer_raises_string_int(self):
        for s in ['`a`', 'g`a`', 'a']:
            with self.subTest(s=s):
                with self.assertRaises(RemesParserException):
                    parse_indexer(tokenize(f'[{s},2]'), 0)

    def test_parse_indexer_raises_string_slicer(self):
        for s in ['`a`', 'g`a`', 'a']:
            with self.subTest(s=s):
                with self.assertRaises(RemesParserException):
                    parse_indexer(tokenize(f'[{s},2:]'), 0)

    def test_parse_indexer_raises_binop(self):
        with self.assertRaises(RemesParserException):
            parse_indexer(tokenize('[a,in]'), 0)

    def test_parse_indexer_boolean_index_obj(self):
        query = tokenize("[@>=3]")
        js = {'a':1,'b':2,'c':4}
        correct_out = {'a':False,'b':False,'c':True}
        self.assertEqual(parse_indexer(query, 0)[0]['value'](js), correct_out)

    def test_parse_indexer_boolean_index_array(self):
        query = tokenize("[@>=3]")
        js = [1,2,4]
        correct_out = [False,False,True]
        self.assertEqual(parse_indexer(query, 0)[0]['value'](js), correct_out)

    ###################
    ## vectorized operations tests
    ###################

    def test_boolean_json_scalar_arr(self):
        self.assertEqual(binop_json_scalar(BINOPS['<'][0], [1, 2], 2),
                        [True, False])

    def test_boolean_scalar_json_arr(self):
        self.assertEqual(binop_scalar_json(BINOPS['>'][0], 2, [1, 2]),
                        [True, False])

    def test_boolean_json_scalar_dict(self):
        self.assertEqual(binop_json_scalar(BINOPS['<'][0],
                                           {'a': 1, 'b': 2}, 2),
                         {'a': True, 'b': False})

    def test_boolean_scalar_json_dict(self):
        self.assertEqual(binop_scalar_json(BINOPS['>='][0], 1.5,
                                           {'a': 1, 'b': 2}),
                         {'a': True, 'b': False})

    def test_regex_index(self):
        self.assertEqual(apply_regex_index({'ab': 1, 'ba': 2, 'c': 3}, re.compile('^a')), {'ab': 1})

    def test_multi_index_regex_str(self):
        self.assertEqual(apply_multi_index([re.compile('^a'), 'c'])({'ab': 1, 'ba': 2, 'c': 3}), {'ab': 1, 'c': 3})

    def test_multi_index_only_str(self):
        self.assertEqual(apply_multi_index(['a', 'c'])({'ab': 1, 'ba': 2, 'c': 3}), {'c': 3})

    def test_multi_index_one_int_arr(self):
        self.assertEqual(apply_multi_index([1])([1,2,3]), [2])

    def test_multi_index_two_int_arr(self):
        self.assertEqual(apply_multi_index([0,2])([1,2,3]), [1,3])

    def test_multi_index_slice_int_arr(self):
        self.assertEqual(apply_multi_index([0, slice(2, None)])([1,2,3]), [1, 3])

    def test_multi_index_slice_arr_int(self):
        self.assertEqual(apply_multi_index([slice(2, None), 0])([1,2,3]), [3, 1])

    def test_binop_two_jsons_arrs(self):
        x = [1,2,3]
        for func in [operator.add, operator.sub, operator.mul, pow]:
            with self.subTest(func=func):
                self.assertEqual(binop_two_jsons(func, x, x),
                                [func(e, e) for e in x])

    def test_binop_two_jsons_objs(self):
        x = {'a': 1, 'b': 2}
        for func in [operator.add, operator.sub, operator.mul, pow]:
            with self.subTest(func=func):
                self.assertEqual(binop_two_jsons(func, x, x),
                                {k: func(v, v) for k,v in x.items()})

    def test_has_pattern(self):
        self.assertTrue(BINOPS['=~'][0]('abc', 'a'), True)

    def test_binop_json_scalar_arr(self):
        x = ['abc', 'bcd']
        for func in [operator.add, has_pattern]:
            with self.subTest(func=func):
                self.assertEqual(binop_json_scalar(func, x, 'a'),
                                 [func(e, 'a') for e in x])

    def test_binop_scalar_json_arr(self):
        x = ['abc', 'bcd']
        for func in [operator.add, has_pattern]:
            with self.subTest(func=func):
                self.assertEqual(binop_scalar_json(func, 'a', x),
                                 [func('a', e) for e in x])

    def test_binop_json_scalar_obj(self):
        x = {'a': 'abc', 'b': 'bcd'}
        for func in [operator.add, has_pattern]:
            with self.subTest(func=func):
                self.assertEqual(binop_json_scalar(func, x, 'a'),
                                 {k: func(e, 'a') for k, e in x.items()})

    def test_binop_scalar_json_obj(self):
        x = {'a': 'abc', 'b': 'bcd'}
        for func in [operator.add, has_pattern]:
            with self.subTest(func=func):
                self.assertEqual(binop_scalar_json(func, 'a', x),
                                 {k: func('a', e) for k, e in x.items()})

    def test_resolve_binop_scalar_json(self):
        self.assertEqual(resolve_binop(operator.add, int_node(2), expr([1,2,3])), expr([3, 4, 5]))

    def test_resolve_binop_json_scalar(self):
        self.assertEqual(resolve_binop(operator.sub, expr([1,2,3]), int_node(2)), expr([-1,0,1]))

    def test_resolve_binop_two_jsons(self):
        self.assertEqual(resolve_binop(pow, expr([1,2,3]), expr([1,2,3])), expr([1, 4, 27]))

    def test_resolve_binop_two_scalars(self):
        self.assertEqual(resolve_binop(pow, int_node(2), int_node(3)), int_node(8))

    ##############
    ## parse_arg_function tests
    ##############

    def test_parse_arg_function_s_len(self):
        test_parse_arg_function(self, 's_len', '(`abcd`)', [], int_node(4))

    def test_parse_arg_function_too_many_args(self):
        with self.assertRaises(RemesParserException):
            parse_arg_function(tokenize('(`abc`, 3)'),  0, arg_function(FUNCTIONS['s_len']))

    def test_parse_arg_function_too_few_args(self):
        with self.assertRaises(RemesParserException):
            parse_arg_function(tokenize('(`abc`, `a`)'), 0, arg_function(FUNCTIONS['s_sub']))

    def test_parse_arg_function_wrong_type_mandatory_arg(self):
        # second arg should be a string
        with self.assertRaises(RemesParserException):
            parse_arg_function(tokenize('(`abc`, 3, 4)'), 0, arg_function(FUNCTIONS['s_find']))

    def test_parse_arg_function_wrong_type_optional_arg(self):
        # third arg should be an int
        with self.assertRaises(RemesParserException):
            parse_arg_function(tokenize('(`abc`, `a`, `a`)'), 0, arg_function(FUNCTIONS['s_find']))

    def test_parse_arg_function_optional_arg_missing_vec(self):
        test_parse_arg_function(self, 'round', '(j`[1.0,2.0,3.0]`)', [], expr([1,2,3]))

    def test_parse_arg_function_optional_arg_missing_scalar(self):
        test_parse_arg_function(self, 'round', '(23.56)', [], num(24))

    def test_parse_arg_function_arg_function_arg(self):
        test_parse_arg_function(self, 'int', '(float(`2.4`))', [], int_node(2))

    def test_parse_arg_function_scalar_arg_function_args(self):
        test_parse_arg_function(self, 'round', '(23.567, s_len(`2b`))', [], num(23.57))
        
    def test_parse_arg_function_non_vectorized(self):
        test_parse_arg_function(self, 'sum', '(ifelse(j`[true,false,true]`, 2, 1))', [],  num(5))

    def test_parse_arg_function_non_vectorized_optional_args(self):
        test_parse_arg_function(self, 'irange', '(2, 4)', [], expr([2,3]))

    def test_s_join(self):
        test_parse_arg_function(self, 's_join', '(`_`, @)', ['1', '2'], '1_2')

    def test_flatten(self):
        test_parse_arg_function(self, 'flatten', '(@)', [[1], [2]], [1,2])
        
    def test_flatten_2(self):
        test_parse_arg_function(self, 'flatten', '(@, 2)', [[[1]], [[2], 3]], [1,2,3])
        
    ##############
    ## parse_expr_or_scalar_func tests
    ##############
    def test_parse_expr_or_scalar_func_paren_wrapped_int(self):
        test_parse_expr_or_scalar_func(self, '(((1)))', [], int_node(1))

    def test_parse_expr_or_scalar_func_s_slice(self):
        test_parse_expr_or_scalar_func(self, 's_slice(`abc`, 0)', [], string('a'))

    def test_parse_expr_or_scalar_func_paren_wrapped_arg_function(self):
        test_parse_expr_or_scalar_func(self, '(s_len((`a`)))', [], int_node(1))

    def test_parse_expr_or_scalar_func_paren_imbalanced_parens(self):
        for testcase in [
            f'((2)',
            f')2(',
            f'))'
        ]:
            with self.subTest(testcase=testcase):
                with self.assertRaises(RemesParserException):
                    parse_expr_or_scalar_func(tokenize(testcase), 0)

    #############
    ## handling binops
    #############
    def test_parse_expr_or_scalar_func_simple_binops(self):
        test_parse_expr_or_scalar_func(self, '@ + 3', [1, 2], [4, 5])

    def test_parse_expr_or_scalar_func_binop_uminus_first_arg(self):
        test_parse_expr_or_scalar_func(self, '-2*3', [], int_node(-6))

    def test_parse_expr_or_scalar_func_uminus_second_arg(self):
        test_parse_expr_or_scalar_func(self, '2*-3', [], int_node(-6))

    def test_parse_expr_or_scalar_func_nested_binops(self):
        test_parse_expr_or_scalar_func(self, '4 - -2**2 - 3/4', [], num(7.25))

    def test_parse_expr_or_scalar_func_parens_first_arg(self):
        test_parse_expr_or_scalar_func(self, '(3 - 2)*3', [], int_node(3))

    def test_parse_expr_or_scalar_func_parens_second_arg(self):
        test_parse_expr_or_scalar_func(self, '3/(2-4)', [], num(-1.5))

    def test_parse_expr_or_scalar_func_parens_not_first_or_last(self):
        test_parse_expr_or_scalar_func(self, '2**-(1--2)-5', [], num(2**-(1--2)-5))

    ###############
    ## apply_indexer_list
    ###############
    def test_apply_indexer_list_one_varname_list(self):
        test_apply_indexer_list(self, ['[a,b]'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2})

    def test_apply_indexer_list_one_slicer_list(self):
        test_apply_indexer_list(self, ['[0,3]'], [1,2,3,4], [1,4])

    def test_apply_indexer_list_slicer_then_varname(self):
        test_apply_indexer_list(self, ['[1]', '.a'], [{'a':1, 'b': 2, 'c': 3}, {'a': 4, 'b': 5, 'c': 6}], [{'a': 4}])

    def test_apply_indexer_list_three_slicer_lists(self):
        obj = [[[ 0,  1,  2],
                [ 3,  4,  5]],

               [[ 6,  7,  8],
                [ 9, 10, 11]],

               [[12, 13, 14],
                [15, 16, 17]],

               [[18, 19, 20],
                [21, 22, 23]]]
        correct_out = [
            [[3,   4]],
            [[9,  10]],
            [[21, 22]]
        ]
        test_apply_indexer_list(self, ["[0,1,3]", "[1:]", "[0, 1]"], obj, correct_out)

    def test_apply_indexer_list_two_varname_lists(self):
        obj = {'a': {'b': 1, 'c': 2, 'd': 3}, 'b': {'b': 1, 'c': 2, 'd': 3}, 'e': 'foobar'}
        correct_out = {'a': {'d': 3}, 'b': {'d': 3}}
        test_apply_indexer_list(self, ["[a,b]", ".d"], obj, correct_out)

    def test_apply_indexer_list_slicers_varnames_slicers(self):
        obj = [{'name': 'foo',
  'players': [{'name': 'alice', 'hits': [3, 4, 2, 5]}]},
 {'name': 'bar',
  'players': [{'name': 'carol', 'hits': [7, 3, 0, 5]},
   {'name': 'dave', 'hits': [1, 0, 4, 10]}]}]
        correct_out = [{'players': [{'hits': [1, 0, 4, 10], 'name': 'dave'}]}]
        test_apply_indexer_list(self, ["[:]", ".players", "[1]"], obj, correct_out)

    ###############
    ## indexing in parse_expr_or_scalar_func
    ###############
    def test_parse_expr_or_scalar_index_obj(self):
        test_parse_expr_or_scalar_func(self, '@.foo', {'foo': 1}, 1)

    def test_parse_expr_or_scalar_index_arr(self):
        test_parse_expr_or_scalar_func(self, '@[0]', [1,2], 1)

    def test_parse_expr_or_scalar_index_arr_obj(self):
        test_parse_expr_or_scalar_func(self, '@[0].foo', [{'foo': 1, 'bar': 2},{'foo': 'a', 'bar': 'b'}], 1)

    def test_parse_expr_or_scalar_index_obj_arr(self):
        test_parse_expr_or_scalar_func(self, '@[foo,bar][1:]', {'foo': [1,'a'], 'bar': [2, 'b']}, {'foo': ['a'], 'bar': ['b']})


    ################
    ## search tests with functions
    ################
    def test_search_sort_by_key(self):
        self.assertEqual(search("sort_by(@, foo)", [{'foo': 2, 'bar': 1}, {'foo': 1, 'bar': 3}]), [{'foo': 1, 'bar': 3}, {'foo': 2, 'bar': 1}])
        
    def test_search_sort_by_key_reverse(self):
        self.assertEqual(search("sort_by(@, a, true)", [{'a': 1, 'b': 2}, {'a': 2, 'b': 1}]), [{'a': 2, 'b': 1}, {'a': 1, 'b': 2}])
        
    def test_search_max_by_key(self):
        self.assertEqual(search("max_by(@, foo)", [{'foo': 2, 'bar': 1}, {'foo': 1, 'bar': 3}]), {'foo': 2, 'bar': 1})
        
    def test_search_min_by_key(self):
        self.assertEqual(search("min_by(@, foo)", [{'foo': 2, 'bar': 1}, {'foo': 1, 'bar': 3}]), {'foo': 1, 'bar': 3})
        
    def test_search_max_by_idx(self):
        self.assertEqual(search('max_by(@, 0)', [[1, 2], [3, 0]]), [3, 0])
        
    def test_search_min_by_idx(self):
        self.assertEqual(search('min_by(@, 0)', [[1, 2], [3, 0]]), [1, 2])
        
    def test_search_sort_by_idx(self):
        self.assertEqual(search('sort_by(@, 0, true)', [[1, 2], [3, 0]]), [[3, 0], [1, 2]])
        
    def test_search_is_str_num_expr(self):
        inputs = ["`a`", "1", "j`[[2]]`", "-1.0", "true"]
        for func, correct_results in [
            ['is_str', [True, False, [False], False, False]],
            ['is_num', [False, True, [False], True, True]], 
            ['is_expr', [False, False, [True], False, False]],
        ]:
            for input_, correct_result in zip(inputs, correct_results):
                with self.subTest(input_=input_, func=func, correct_result=correct_result):
                    self.assertEqual(search(f"{func}({input_})", []), correct_result)
                    
    def test_search_unique(self):
        self.assertEqual(set(search("unique(@)", [1,1,2,3,1,3,3])), {1,2,3})
        
    def test_search_unique_sorted(self):
        self.assertEqual(search("unique(@)", [1,1,2,3,1,3,3]), [1,2,3])
        
    def test_search_value_counts(self):
        self.assertEqual(search("value_counts(@)", list('abca')), {'a': 2, 'b': 1, 'c': 1})
        
    def test_search_s_sub(self):
        self.assertEqual(search("s_sub(@, `a`, `b`)", ["a bad dog"]), ["b bbd dog"])
        
    def test_search_s_sub_regex(self):
        self.assertEqual(search("s_sub(@, g`(?i)a`, `b`)", ["A bad dog"]), ["b bbd dog"])
        
    def test_search_len_function(self):
        self.assertEqual(search("len(@)", [1,2,3]), 3)

    def test_search_len_function_j_json(self):
        self.assertEqual(search("len(j`[1,2,3]`)", []), 3)
        
    def test_search_s_slice(self):
        self.assertEqual(search('s_slice(`abc`, ::2)', []), 'ac')

    def test_search_scalar_argfunc_with_cur_js_argfunc_arg(self):
        self.assertEqual(search("str(len(@))", [1,2,3]), '3')
        
    def test_search_iterable_argfunc_with_cur_js_argfunc_arg(self):
        self.assertEqual(search("sorted(values(@))", {'foo': 2, 'bar': 1}), [1, 2])
    
    def test_search_isna(self):
        self.assertEqual(search("isna(@)", [1, float('nan')]), [False, True])
    
    
    ################
    ## search tests with binops
    ################
    def test_search_curdoc_binop_curdoc(self):
        self.assertEqual(search("@ ** @", [1,2]), [1,4])
        
    def test_search_uminus_curdoc(self):
        self.assertEqual(search("-@", [1]), [-1])
        
    def test_search_idx_binop(self):
        self.assertEqual(search("@[:].bar < 3", [{'foo': 2, 'bar': 1}, {'foo': 1, 'bar': 3}]), [True, False])
        
    ###############
    ## search tests with indexing
    ###############
    def test_search_dot(self):
        self.assertEqual(search("@.foo", {'foo': 1}), 1)
        
    def test_search_dot_j_json(self):
        self.assertEqual(search('j`{"foo": 1}`.foo', []), 1)

    def test_search_dot_after_bool_idx(self):
        syntax_options = [
        "@[:][s_slice(@.name, 0) == `f`].x",
        # this is a reasonably non-stupid way to do the same thing as
        # the first. If both fail, we've got a problem
        "@[s_slice(@[:].name, 0) == `f`].x"
        ]
        x = [{'name': 'a', 'x': 1}, {'name': 'fo', 'x': 2}, {'name': 'fb', 'x': 3}]
        correct_out = [2, 3]
        mismatches = ''
        for syntax in syntax_options:
            try:
                result = search(syntax, x)
                if result != correct_out:
                    mismatches += f'with query {syntax}, {result} != {correct_out}'
            except Exception as ex:
                mismatches += f'with query {syntax}, {str(ex)} != {correct_out}\n'
        self.assertTrue(bool(mismatches), msg=mismatches)
                        
    def test_search_idx_after_bool_idx(self):
        syntax_options = [
            "@[@[:].foo > 1][0]", 
        # as with the above alt_syntax test, option 2 is kludgy but OK
        # as an alternative to the syntax I would prefer to work
            "(@[@[:].foo > 1])[0]"
        ]
        x = [{'foo': 1, 'bar': 3}, {'foo': 2, 'bar': 1}]
        correct_out = {'foo': 2, 'bar': 1}
        mismatches = ''
        for syntax in syntax_options:
            try:
                result = search(syntax, x)
                if result != correct_out:
                    mismatches += f'with query {syntax}, {result} != {correct_out}'
            except Exception as ex:
                mismatches += f'with query {syntax}, {str(ex)} != {correct_out}\n'
        self.assertTrue(bool(mismatches), msg=mismatches)
                        
    def test_search_dot_then_slice(self):
        self.assertEqual(search("@.foo[0]", {'foo': [1]}), 1)
        
    def test_search_dot_then_slice_j_json(self):
        self.assertEqual(search('j`{"foo": [1,2]}`.foo[:1]', []), [1])
        
    def test_search_dot_dot(self):
        self.assertEqual(search("@.foo.bar", {'foo': {'bar': 1}}), 1)
        
    def test_search_dot_dot_j_json(self):
        self.assertEqual(search('j`{"foo": {"bar": 1}}`.foo.bar', []), 1)
        
    def test_search_bool_idx_arr(self):
        self.assertEqual(search("@[@ >= 2]", [1,2,3]), [2, 3])
        
    def test_search_bool_idx_dict(self):
        self.assertEqual(search("@[@ > 2]", {'a': 1, 'b': 3}), {'b': 3})
        
    def test_search_bool_idx_j_json(self):
        self.assertEqual(search("j`[1,2,3]`[j`[1,2,3]` >= 2]", []), [2, 3])
        
    def test_search_idx_on_expr_function(self):
        self.assertEqual(search("sorted(@)[0]", [2, 1]), 1)
    
    def test_search_idx_curdoc_bool_idx(self):
        syntax_options = [
            "@[:][@.foo == 1]", 
            # as with the above alt_syntax test, option 2 is kludgy but OK
            # as an alternative to the syntax I would prefer to work
            "@[@[:].foo == 1]"
        ]
        x = [{'foo': 1, 'bar': 3}, {'foo': 2, 'bar': 1}]
        correct_out = [{'foo': 1, 'bar': 3}]
        mismatches = ''
        for syntax in syntax_options:
            try:
                result = search(syntax, x)
                if result != correct_out:
                    mismatches += f'with query {syntax}, {result} != {correct_out}'
            except Exception as ex:
                mismatches += f'with query {syntax}, {str(ex)} != {correct_out}\n'
        self.assertTrue(bool(mismatches), msg=mismatches)
        
    def test_search_arg_function_scalar_query_result(self):
        self.assertEqual(search("str(@.foo)", {'foo': 1}), '1')
        
    def test_search_dot_parens_idx(self):
        self.assertEqual(search("(@.foo)[0]", {'foo': [1,2], 'bar': 'a'}), 1)
        
    def test_search_dot_parens_dot(self):
        self.assertEqual(search('(@.foo).bar', {'foo': {'bar': 1}}), 1)
        
    def test_search_idx_parens_dot(self):
        self.assertEqual(search('(@[0]).a', [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]), 2)
        
    def test_search_idx_parens_idx(self):
        self.assertEqual(search('(@[:1])[1:]', [[1,2],[3,4]]), [[2]])
        
    ###############
    ## projections
    ###############
    def test_projection_obj(self):
        self.assertEqual(search("@{foo: @[0], bar: @[1]}", [1, 2]), {'foo': 1, 'bar': 2})
        
    def test_projection_arr(self):
        self.assertEqual(search("@{sum(@.hits), @.foo}", {'foo': 'bar', 'hits': [1,2,3]}), [6, 'bar'])
        
    def test_projection_arr_after_bool_idx(self):
        self.assertEqual(search("@[@[:].foo == 1]{@.foo, @.bar}", [{'foo': 1, 'bar': 3}, {'foo': 2, 'bar': 1}]), [[1, 3]])
        
    def test_projection_arr_after_idx(self):
        self.assertEqual(search("@[:]{@.foo, @.bar}", [{'foo': 1, 'bar': 3}, {'foo': 2, 'bar': 1}]), [[1, 3], [2, 1]])
        
    def test_projection_arr_before_idx(self):
        self.assertEqual(search("@[:]{@.foo, @.bar}[1:]", [{'foo': 1, 'bar': 3}, {'foo': 2, 'bar': 1}]), [[3], [1]])
        
    def test_arg_func_projection_arr(self):
        self.assertEqual(search("len(@{@.foo * @.bar})", {'foo': 1, 'bar': 2}), 1)
        
    def test_projection_arr_parens_then_idx(self):
        self.assertEqual(search("(@[:]{@.foo, @.bar})[1]", [{'foo': 1, 'bar': 3}, {'foo': 2, 'bar': 1}]), [2, 1])
        
    def test_projection_obj_then_dot(self):
        self.assertEqual(search("@{foo: @[0], bar: @[1]}.foo", [1, 2]), 1)
        
    def test_projection_obj_parens_then_dot(self):
        self.assertEqual(search("(@{foo: @[0], bar: @[1]}).foo", [1,2]), 1)
        

if __name__ == '__main__':
    unittest.main()
