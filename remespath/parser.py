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
import json
import operator
import re
import unittest
import logging

logging.basicConfig(level=logging.DEBUG)
RemesPathLogger = logging.getLogger(name='RemesPathLogger')


NUM_START_CHARS = set('-0123456789')
num_match = num_regex = re.compile(("(-?(?:0|[1-9]\d*))"   # any int or 0
                "(\.\d+)?" # optional decimal point and digits after
                "([eE][-+]?\d+)?" # optional scientific notation
                )).match
quoted_string_match = re.compile((
    r'((?:[^`]' # any number of (non-backtick characters...
    r'|(?<=\\)`)*)' # or backticks preceded by a backslash)
    r'(?<!\\)`')).match # terminated by a backtick not preceded by a backslash
int_match = re.compile('(-?\d+)').match

unquoted_string_match = re.compile("[a-zA-Z_][a-zA-Z_0-9]*").match

EXPR_FUNC_ENDERS = {']', ':', '}', ',', ')'}
# these tokens have high enough precedence to stop an expr_function or scalar_function


class RemesParserException(Exception):
    def __init__(self, message, ind=None, x=None):
        self.x = x
        self.ind = ind
        self.message = message

    def __str__(self):
        if self.ind is None and self.x is None:
            return self.message
        if len(self.x) > 15:
            if self.ind < 11:
                xrepr = self.x[:15] + '...'
                space_before_caret = self.ind
            elif len(self.x) - self.ind <= 8:
                xrepr = '...' + self.x[self.ind-8:]
                space_before_caret = 11
            else:
                xrepr = '...' + self.x[self.ind-8:self.ind+8] + '...'
                space_before_caret = 11
        else:
            xrepr = self.x
            space_before_caret = self.ind
        return f'{self.message} (position {self.ind})\n{xrepr}\n{self.x[self.ind]}' #{space_before_caret*" "}^'


class VectorizedArithmeticException(Exception):
    pass


##############
## INDEXER FUNCTIONS
##############

def apply_multi_index(x, inds):
    '''x: a list or dict
inds: a list of indices or keys to select'''
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
            else:
                # k is a slice
                out.extend(x[k])
        return out


def apply_boolean_index(x, inds):
    if len(inds) != len(x):
        raise VectorizedArithmeticException(f"Boolean index length ({len(inds)}) != object/array length ({len(x)})")
    if not all(isinstance(e, bool) for e in inds):
        raise VectorizedArithmeticException('boolean index contains non-booleans')
    if isinstance(x, dict):
        try:
            return {k: v for k, v in x.items() if inds[k]}
        except KeyError as ex:
            raise VectorizedArithmeticException(str(ex))
    return [v for v, ind in zip(x, inds) if ind]


def apply_regex_index(obj, regex):
    return {k: v for k, v in obj.items() if regex.search(k)}


def apply_indexer_list(obj, indexers):
    '''recursively search obj to match indexers'''
    idxr, _ = parse_indexer(indexers[0], expr(obj), 0)
    idx_func = None
    children = idxr['children']
    if idxr['type'][-5:] == '_list':
        idx_func = lambda x: apply_multi_index(x, children)
    else:
        idx_func = lambda x: apply_boolean_index(x, children)
        if not isinstance(children, (list, dict)):
            return obj if children else None
    result = idx_func(obj)
    RemesPathLogger.info(f"In apply_indexer_list, {idxr = }, {obj = }, {result = }")
    is_dict = isinstance(obj, dict)
    has_one_option = False
    k = None
    if len(children) == 1 and isinstance(children[0], (int, str)):
        has_one_option = True
        k = 0 if not is_dict else children[0]
    if len(indexers) == 1:
        if result and has_one_option:
            return result[k]
        return result
    if is_dict:
        if has_one_option:
            # don't need to specify the key when only one key is possible
            return apply_indexer_list(result[k], indexers[1:])
        out = {}
        for k, v in result.items():
            subdex = apply_indexer_list(v, indexers[1:])
            if subdex:
                out[k] = subdex
    else:
        if has_one_option:
            # don't need an array output when you're getting a single index
            return apply_indexer_list(result[0], indexers[1:])
        out = []
        for v in idx_func(obj):
            subdex = apply_indexer_list(v, indexers[1:])
            if subdex:
                out.append(subdex)
    RemesPathLogger.info(f"apply_indexer_list returns {result}")
    return out


###############
## VECTORIZED STUFF
###############

def binop_two_jsons(func, a, b):
    la, lb = len(a), len(b)
    if la !=  lb :
        raise VectorizedArithmeticException(f"Tried to add two objects of lengths {la} and {lb}")
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


def apply_arg_function(func, out_types, *args):
    x = args[0]
    other_args = [None if x is None else x['value'] for x in args[1:]]
    xval = x['value']
    xtype = x['type']
    out_type = None
    if isinstance(out_types, list):
        out_type = out_types[0] if xtype == 'expr' else out_types[1]
    elif out_types == '?':
        out_type = xtype
    else:
        out_type = out_types
    ast_tok_builder = AST_TOK_BUILDER_MAP[out_type]
    if x['type'] == 'expr':
        if isinstance(xval, dict):
            return ast_tok_builder({k: func(v, *other_args) for v in xval.items()})
        elif isinstance(xval, list):
            return ast_tok_builder([func(v, *other_args) for v in xval])
    return ast_tok_builder(func(xval, *other_args))


################
## PARSER FUNCTIONS
################

def peek_next_token(x, ii):
    if ii + 1 >= len(x):
        return ''
    return x[ii + 1]


def parse_quoted_string(s, string_type_modifier=''):
    mtch = quoted_string_match(s)
    try:
        out = mtch.groups()[0].replace('\\`', '`')
    except:
        raise RemesParserException("Unterminated string literal")
    RemesPathLogger.debug(f'{s = }')
    if string_type_modifier == 'json':
        try:
            return expr(json.loads(out))
        except json.JSONDecodeError as ex:
            raise RemesParserException(f"Malformed json ({ex})")
    elif string_type_modifier == 'regex':
        try:
            return regex(re.compile(out))
        except Exception as ex:
            raise RemesParserException(f"Malformed regex ({ex})")
    return quoted_string(out)


def parse_unquoted_string(s):
    '''parse any unquoted string. This includes function names, names of
special constants like true and NaN, and names of keys in iterables.
Thus, this can return any of the following AST node types: expr_function, binop, bool, null, num, or (most commonly) unquoted_string.
    '''
    if s in FUNCTIONS:
        return expr_function(*FUNCTIONS[s])
    if s in BINOPS:
        return binop_function(BINOPS[s])
    if s in CONSTANTS:
        val = CONSTANTS[s]
        if isinstance(val, bool):
            return bool_node(val)
        if val is None:
            return null_node()
        else:
            # remaining special cases are NaN and +/-Infinity, both floats
            return num(val)
    return unquoted_string(s)


def parse_num(query, uminus):
    multiplier = -1 if uminus else 1
    mtch = num_match(query)
    grps = mtch.groups()
    if not (grps[1] or grps[2]):
        return int_node(int(grps[0]) * multiplier)
    elif grps[1]:
        new_obj = float(grps[0] + grps[1] if not grps[2] else ''.join(grps))
    else:
        new_obj = float(grps[0] + grps[2])
    return num(new_obj * multiplier)


def parse_slicer(query, jsnode, ii, first_num):
    grps = []
    last_num = first_num
    end = ii
    while end < len(query):
        RemesPathLogger.debug(f'in parse_slicer, {query[end] = }, {grps = }')
        if query[end] == ':':
            grps.append(last_num)
            last_num = None
            end += 1
            continue
        if query[end][0] not in NUM_START_CHARS:
            break
        try:
            numtok, end = parse_expr_or_scalar_func(query, jsnode, end)
            assert numtok['type'] == 'int'
            last_num = numtok['value']
        except:
            raise RemesParserException("Found non-integer while parsing slicer", ii, query)
        if len(grps) == 2:
            break
    grps.append(last_num)
    RemesPathLogger.debug(f'at parse_slicer return, {query = }, {end = }, {grps = }')
    return slicer(grps), end - 1


def parse_indexer(query, jsnode, ii):
    last_tok = None
    t = query[ii]
    if t == '.':
        nt = peek_next_token(query, ii)
        if not nt:
            raise RemesParserException("Expected token after '.'", ii, query)
        if nt[:2] == 'g`':
            last_tok = parse_quoted_string(nt[2:], 'regex')
        elif nt[0] == '`':
            last_tok = parse_quoted_string(nt[1:])
        elif re.match('[a-zA-Z_]', nt):
            try:
                last_tok = parse_unquoted_string(nt)
                assert last_tok['type'] == 'unquoted_string'
            except Exception as ex:
                raise RemesParserException('Indexer starting with "." must have exactly one varname', ii, query)
        else:
            raise RemesParserException('Indexer starting with "." must have exactly one varname', ii, query)
        return varname_list([last_tok['value']]), ii + 1
    elif query[ii] != '[':
        raise RemesParserException('Indexer must start with "." or "["', ii, query)
    children = []
    indexer = None
    ii += 1
    while ii < len(query):
        t = query[ii]
        c = t[0]
        RemesPathLogger.debug(f'in parse_indexer, {last_tok = }, {c = }, {ii = }')
        if c == ']':
            children.append(last_tok['value'])
            if not indexer:
                if last_tok['type'] in VARNAME_SUBTYPES:
                    indexer = varname_list(children)
                elif last_tok['type'] in ['slicer', 'int']:
                    indexer = slicer_list(children)
                else:
                    indexer = boolean_index(last_tok['value'])
            if (indexer['type'] == 'slicer_list' and last_tok['type'] not in SLICER_SUBTYPES) \
            or (indexer['type'] == 'varname_list' and last_tok['type'] not in VARNAME_SUBTYPES):
                raise RemesParserException("Cannot have indexers with a mix of ints/slicers and strings", ii, query)
            return indexer, ii + 1
        elif last_tok and c != ',' and c != ':':
            raise RemesParserException("Consecutive indexers must be separated by commas", ii, query)
        elif c == ',':
            # figure out how to deal with slices
            # if not all(x is None
            if not indexer:
                if last_tok['type'] in VARNAME_SUBTYPES:
                    indexer = varname_list([])
                elif last_tok['type'] in SLICER_SUBTYPES:
                    indexer = slicer_list([])
                children = indexer['children']
            elif (indexer['type'] == 'slicer_list' and last_tok['type'] not in SLICER_SUBTYPES) \
            or (indexer['type'] == 'varname_list' and last_tok['type'] not in VARNAME_SUBTYPES):
                raise RemesParserException("Cannot have indexers with a mix of ints/slicers and strings", ii, query)
            children.append(last_tok['value'])
            last_tok = None
            ii += 1
        elif c == ':':
            if last_tok is None:
                last_tok, ii = parse_slicer(query, jsnode, ii, None)
            elif last_tok['type'] == 'int':
                last_tok, ii = parse_slicer(query, jsnode, ii, last_tok['value'])
            else:
                raise RemesParserException(f"Expected token other than ':' after {last_tok} in an indexer", ii, query)
            ii += 1
        else:
            last_tok, ii = parse_expr_or_scalar_func(query, jsnode, ii)
    raise RemesParserException("Unterminated indexer (EOF)")


def resolve_binop(binop, a, b):
    '''apply a binop to two args that are exprs or scalars'''
    RemesPathLogger.info(f"In resolve_binop, {binop = }, {a = }, {b = }")
    aval, bval = a['value'], b['value']
    atype, btype = a['type'], b['type']
    if atype == 'expr':
        if btype == 'expr':
            return expr(binop_two_jsons(binop, aval, bval))
        return expr(binop_json_scalar(binop, aval, bval))
    if btype == 'expr':
        return expr(binop_scalar_json(binop, aval, bval))
    outval = binop(aval, bval)
    return AST_TYPE_BUILDER_MAP[type(outval)](outval)


def resolve_binop_tree(binop, a, b):
    '''applies a binop to two args that are exprs, scalars, or other binops'''
    RemesPathLogger.info(f"In resolve_binop_tree, {binop = }, {a = }, {b = }")
    if a['type'] == 'binop':
        a = resolve_binop_tree(a['value'][0], *a['children'])
    if b['type'] == 'binop':
        b = resolve_binop_tree(b['value'][0], *b['children'])
    return resolve_binop(binop, a, b)


def parse_expr_or_scalar(query, jsnode, ii):
    '''Returns any single scalar or expr, including arg functions.
Does not resolve binops.'''
    if not query:
        raise RemesParserException("Empty query")
    t = query[ii]
    c = t[0]
    RemesPathLogger.info(f"in parse_expr_or_scalar, {t = }, {ii = }")
    if c == '@':
        last_tok = jsnode
        ii += 1
    elif c == '-':
        last_tok, ii = parse_arg_function(query, jsnode, ii, '-')
    elif c in NUM_START_CHARS:
        last_tok = parse_num(t, False)
        ii += 1
    elif c == '`':
        last_tok = parse_quoted_string(t[1:], '')
        ii += 1
    elif c == '(':
        unclosed_parens = 1
        subquery = []
        for end in range(ii + 1, len(query)):
            subtok = query[end]
            if subtok == '(':
                unclosed_parens += 1
            elif subtok == ')':
                unclosed_parens -= 1
                if unclosed_parens == 0:
                    last_tok, subii = parse_expr_or_scalar_func(subquery, jsnode, 0)
                    ii = end + 1
                    break
            subquery.append(subtok)
        if unclosed_parens:
            raise RemesParserException("Unmatched '('", ii, query)
    else:
        if not re.match('[a-zA-Z_]', c):
            raise RemesParserException(f"Char '{c}' can't begin an expr or scalar", ii, query)
        ii += 1
        if t[:2] == 'g`':
            last_tok = parse_quoted_string(t[2:], 'regex')
        elif t[:2] == 'j`':
            last_tok = parse_quoted_string(t[2:], 'json')
        elif t in FUNCTIONS:
            last_tok, ii = parse_arg_function(query, jsnode, ii, t)
        elif t in BINOPS:
            raise RemesParserException(f"Found binop '{t}' without left arg", ii, query)
        else:
            last_tok = parse_unquoted_string(t)
    if last_tok['type'] == 'expr':
        idxrs = []
        cur_idxr = None
        # check if the expr has any indexers
        while True:
            delim = peek_next_token(query, ii - 1)
            if delim == '.':
                cur_idxr = query[ii:ii+2]
                ii += 2
            elif delim == '[':
                cur_idxr = []
                open_sqbk_ct = 1
                while ii < len(query):
                    tok = query[ii]
                    cur_idxr.append(tok)
                    if tok == ']':
                        open_sqbk_ct -= 1
                        if open_sqbk_ct == 0:
                            break
                    ii += 1
                ii += 1
            else:
                break
            RemesPathLogger.info(f"In parse_expr_or_scalar, found indexer {cur_idxr}")
            idxrs.append(cur_idxr)
        if idxrs:
            result = apply_indexer_list(last_tok['value'], idxrs)
            last_tok = AST_TYPE_BUILDER_MAP[type(result)](result)
    RemesPathLogger.info(f"parse_expr_or_scalar returns {(last_tok, ii)}")
    return last_tok, ii


def parse_expr_or_scalar_func(query, jsnode, ii):
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
        t = query[ii]
        if t in EXPR_FUNC_ENDERS:
            if not curtok:
                raise RemesParserException("No expression found where expr or scalar expected", ii, query)
            break
        left_tok = curtok
        left_type = '' if not left_tok else left_tok['type']
        is_binop = BINOPS.get(t)
        if is_binop:
            if left_tok is None or left_tok['type'] == 'binop':
                if t != '-':
                    raise RemesParserException("Binop with invalid left operand", ii, query)
                uminus = not uminus
            else:
                curtok = binop_function(*is_binop)
                children = curtok['children']
                func, precedence = is_binop
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
            left_operand, ii = parse_expr_or_scalar(query, jsnode, ii)
            if uminus:
                nt = '' if ii >= len(query) else query[ii]
                if nt != '**':
                    # apply uminus to the left operand unless pow is coming
                    # up. uminus has higher precedence than other binops
                    left_operand = apply_arg_function(operator.neg, '?', left_operand)
                    uminus = False
            curtok = left_operand
        RemesPathLogger.info(f'In parse_expr_or_scalar_func, {root = }, {t = }, {uminus = }')
    if root:
        branch_children[1] = curtok
        RemesPathLogger.debug(f'parse_expr_or_scalar_func resolves binop:\n{root}')
        left_operand = resolve_binop_tree(root['value'][0], *root['children'])
    RemesPathLogger.debug(f'parse_expr_or_scalar_func returns {left_operand}')
    return left_operand, ii


def parse_arg_function(query, jsnode, ii, funcname):
    '''Handles functions that accept arguments using the "(" arg ("," arg)* ")"
syntax.
This is relevant to both the expr_function and scalar_function parts of the
grammar, because it is agnostic about the types of arguments received
    '''
    argfunc = FUNCTIONS[funcname]
    if query[ii] != '(': # and funcname != '-':
        raise RemesParserException(f"Function '{funcname}' must have parens surrounding arguments", ii, query)
    ii += 1
    func, out_types, min_args, max_args, arg_types, is_vectorized = argfunc
    if argfunc.max_args == inf:
        args = []
    else:
        args = [None for ii in range(argfunc.max_args)]
    arg_num = 0
    cur_arg = None
    while ii < len(query):
        t = query[ii]
        c = t[0]
        type_options = arg_types[arg_num]
        # try to parse the current argument as one of the valid types
        RemesPathLogger.debug(f'In parse_arg_function, {t = }, {funcname = }, {query = }, {ii = }, {type_options = }, {arg_num = }')
        try:
            try:
                cur_arg, ii = parse_expr_or_scalar_func(query, jsnode, ii)
            except:
                cur_arg = None
            if 'slicer' in type_options and query[ii] == ':':
                cur_arg, ii = parse_slicer(query, jsnode, ii, cur_arg)
                ii += 1
            if cur_arg is None or cur_arg['type'] not in type_options:
                raise RemesParserException(f"For arg {arg_num} of function {funcname}, expected argument of a type in {type_options}, instead got type {cur_arg['type']}")
        except Exception as ex:
            raise RemesParserException(f"For arg {arg_num} of function {funcname}, expected argument of a type in {type_options}, instead raised exception:\n{str(ex)}")
        # if funcname == '-':
            # # uminus doesn't require parens around its single argument.
            # RemesPathLogger.debug(f'at return of parse_arg_function, {func = }, {out_types = }, {args = }')
            # return apply_arg_function(func, out_types, [cur_arg]), ii
        c = query[ii]
        if arg_num + 1 < min_args and c != ',':
            raise RemesParserException(f"Expected ',' after argument {arg_num} of function {funcname} ({min_args}-{max_args} args)", ii, query)
        if arg_num + 1 == max_args and c != ')':
            raise RemesParserException(f"Expected ')' after argument {arg_num} of function {funcname} ({min_args}-{max_args} args)", ii, query)
        elif max_args < inf:
            args[arg_num] = cur_arg
        else:
            args.append(cur_arg)
        arg_num += 1
        ii += 1
        if c == ')':
            RemesPathLogger.debug(f'at return of parse_arg_function, {func = }, {out_types = }, {args = }')
            if is_vectorized:
                return apply_arg_function(func, out_types, *args), ii
            out = func(*[None if x is None else x['value'] for x in args])
            ast_converter = AST_TOK_BUILDER_MAP[out_types]
            return ast_converter(out), ii
    raise RemesParserException(f"Expected ')' after argument {arg_num} of function {funcname} ({min_args}-{max_args} args)", ii, query)


def parse_projection(query, jsnode, ii):
    raise NotImplementedError
    if query[ii] != '{':
        raise RemesParserException("Projection must begin with '{'", ii, query)
    children = []
    key, val = None, None
    while ii < len(query):
        c = query[ii]
        if c == '`':
            key, ii = parse_quoted_string(query, ii, '')
            try:
                if query[ii] != ':':
                    raise RemesParserException("Expected ':' between key and value in object_projection", ii, query)
                # parse a token here?
                # xpr, ii = parse_expr_or_scalar_func(query,
            except Exception as ex:
                raise RemesParserException(str(ex), ii, query)


def search(query, obj):
    toks = tokenize(query)
    result = parse_expr_or_scalar_func(toks, expr(obj), 0)
    return result[0]['value']


TYPE_PARSER_MAP = {
    'expr_function': parse_expr_or_scalar_func,
    'scalar_function': parse_expr_or_scalar_func,
    'expr': parse_expr_or_scalar_func,
    'int': parse_num,
    'num': parse_num,
    'regex': parse_quoted_string,
    'quoted_string': parse_quoted_string,
    'slicer': parse_slicer,
    'unquoted_string': parse_unquoted_string,
    'bool': parse_unquoted_string,
}


#################
## TESTING
#################

def test_parse_indexer(tester, x, out):
    idx = parse_indexer(tokenize(x), [], 0)
    tester.assertEqual(out, idx[0]['children'])


class RemesPathTester(unittest.TestCase):
    ##############
    ## misc tests
    ##############
    def test_current_node(self):
        self.assertEqual(parse_expr_or_scalar('@', expr([1]), 0), (expr([1]),1))

    ##############
    ## indexer tests
    ##############
    def test_parse_indexer_one_string(self):
        test_parse_indexer(self, '.foo', ['foo'])

    def test_parse_indexer_dot_int_raises(self):
        with self.assertRaises(RemesParserException):
            parse_indexer('.1', [], 0)

    def test_parse_indexer_dot_slicer_raises(self):
        with self.assertRaises(RemesParserException):
            parse_indexer('.:1', [], 0)

    def test_parse_indexer_dot_opensqbk_raises(self):
        with self.assertRaises(RemesParserException):
            parse_indexer('.[1]', [], 0)

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
        test_parse_indexer(self, '[2:,1]', [slice(2, None), 1])

    def test_parse_indexer_raises_int_string(self):
        # indexers containing a mix of strings and ints are disallowed
        for s in ['`a`', 'g`a`', 'a']:
            with self.subTest(s=s):
                with self.assertRaises(RemesParserException):
                    parse_indexer(tokenize(f'[2,{s}]'), [], 0)

    def test_parse_indexer_raises_slicer_string(self):
        for s in ['`a`', 'g`a`', 'a']:
            with self.subTest(s=s):
                with self.assertRaises(RemesParserException):
                    parse_indexer(tokenize(f'[2:,{s}]'), [], 0)

    def test_parse_indexer_raises_string_int(self):
        for s in ['`a`', 'g`a`', 'a']:
            with self.subTest(s=s):
                with self.assertRaises(RemesParserException):
                    parse_indexer(tokenize(f'[{s},2]'), [], 0)

    def test_parse_indexer_raises_string_slicer(self):
        for s in ['`a`', 'g`a`', 'a']:
            with self.subTest(s=s):
                with self.assertRaises(RemesParserException):
                    parse_indexer(tokenize(f'[{s},2:]'), [], 0)

    def test_parse_indexer_raises_binop(self):
        with self.assertRaises(RemesParserException):
            parse_indexer(tokenize('[a,in]'), [], 0)

    def test_parse_indexer_boolean_index_obj(self):
        query = tokenize("[@>=3]")
        js = expr({'a':1,'b':2,'c':4})
        correct_out = (boolean_index({'a':False,'b':False,'c':True}), 5)
        self.assertEqual(parse_indexer(query, js, 0), correct_out)

    def test_parse_indexer_boolean_index_array(self):
        query = tokenize("[@>=3]")
        js = expr([1,2,4])
        correct_out = (boolean_index([False,False,True]), 5)
        self.assertEqual(parse_indexer(query, js, 0), correct_out)

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
        x = {'ab': 1, 'ba': 2, 'c': 3}
        self.assertEqual(apply_regex_index(x, re.compile('^a')), {'ab': 1})

    def test_multi_index_regex_str(self):
        x = {'ab': 1, 'ba': 2, 'c': 3}
        inds = [re.compile('^a'), 'c']
        self.assertEqual(apply_multi_index(x, inds), {'ab': 1, 'c': 3})

    def test_multi_index_only_str(self):
        x = {'ab': 1, 'ba': 2, 'c': 3}
        inds = ['a', 'c']
        self.assertEqual(apply_multi_index(x, inds), {'c': 3})

    def test_multi_index_one_int_arr(self):
        x = [1,2,3]
        inds = [1]
        self.assertEqual(apply_multi_index(x, inds), [2])

    def test_multi_index_two_int_arr(self):
        x = [1,2,3]
        inds = [0,2]
        self.assertEqual(apply_multi_index(x, inds), [1,3])

    def test_multi_index_slice_int_arr(self):
        x = [1,2,3]
        self.assertEqual(apply_multi_index(x, [0, slice(2, None)]), [1, 3])

    def test_multi_index_slice_arr_int(self):
        self.assertEqual(apply_multi_index([1,2,3],
                         [slice(2, None), 0]), [3, 1])

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
        inp = tokenize('(`abcd`)')
        self.assertEqual(parse_arg_function(inp, [], 0, 's_len'), (int_node(4), 3))

    def test_parse_arg_function_too_many_args(self):
        inp = tokenize('(`abc`, 3)')
        with self.assertRaises(RemesParserException):
            parse_arg_function(inp, [], 0, 's_len')

    def test_parse_arg_function_too_few_args(self):
        inp = tokenize('(`abc`, `a`)')
        with self.assertRaises(RemesParserException):
            parse_arg_function(inp, [], 0, 's_sub')

    def test_parse_arg_function_wrong_type_mandatory_arg(self):
        inp = tokenize('(`abc`, 3, 4)')
        # second arg should be a string
        with self.assertRaises(RemesParserException):
            parse_arg_function(inp, [], 0, 's_find')

    def test_parse_arg_function_wrong_type_optional_arg(self):
        inp = tokenize('(`abc`, `a`, `a`)')
        # third arg should be an int
        with self.assertRaises(RemesParserException):
            parse_arg_function(inp, [], 0, 's_find')

    def test_parse_arg_function_optional_arg_missing_vec(self):
        inp = tokenize('(j`[1.0,2.0,3.0]`)')
        correct_out = (expr([1,2,3]), 3)
        # second arg should be a string
        self.assertEqual(parse_arg_function(inp, [], 0, 'round'), correct_out)

    def test_parse_arg_function_optional_arg_missing_scalar(self):
        inp = tokenize('(`23`)')
        correct_out = (int_node(23), 3)
        # second arg should be a string
        self.assertEqual(parse_arg_function(inp, [], 0, 'int'), correct_out)

    def test_parse_arg_function_arg_function_arg(self):
        inp = tokenize('(float(`23`))')
        correct_out = (int_node(23), 6)
        self.assertEqual(parse_arg_function(inp, [], 0, 'int'), correct_out)

    def test_parse_arg_function_scalar_arg_function_args(self):
        inp = tokenize('(23.567, s_len(`2b`))')
        correct_out = (num(23.57), 8)
        self.assertEqual(parse_arg_function(inp, [], 0, 'round'), correct_out)

    def test_parse_arg_function_non_vectorized(self):
        inp = tokenize('sum(ifelse(j`[true,false,true]`, 1, 0))')
        correct_out = (num(2), 11)
        self.assertEqual(parse_arg_function(inp, [], 1, 'sum'), correct_out)

    def test_parse_arg_function_non_vectorized_optional_args(self):
        inp = tokenize('irange(2, 4)')
        correct_out = (expr([2,3]), 6)
        self.assertEqual(parse_arg_function(inp, [], 1, 'irange'), correct_out)

    def test_s_join(self):
        self.assertEqual(parse_arg_function(tokenize("(`_`, @)"), expr(['1','2']), 0, 's_join'), (quoted_string('1_2'), 5))

    def test_flatten(self):
        self.assertEqual(parse_arg_function(tokenize("(@)"), expr([[1],[2]]), 0, 'flatten'), (expr([1,2]), 3))

    def test_parse_expr_or_scalar_func_s_slice(self):
        inp = tokenize('(`abc`, 0)')
        self.assertEqual(parse_arg_function(inp, [], 0, 's_slice'), (quoted_string('a'), 6))
    ##############
    ## parse_expr_or_scalar_func tests
    ##############
    def test_parse_expr_or_scalar_func_paren_wrapped_int(self):
        inp = tokenize('(((1)))')
        correct_out = (int_node(1), 7)
        self.assertEqual(parse_expr_or_scalar_func(inp, [], 0), correct_out)

    def test_parse_expr_or_scalar_func_s_slice(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize('s_slice(`abc`, 0)'), [], 0), (quoted_string('a'), 6))

    def test_parse_expr_or_scalar_func_paren_wrapped_arg_function(self):
        inp = tokenize('(s_len((`a`)))')
        correct_out = (int_node(1), 8)
        self.assertEqual(parse_expr_or_scalar_func(inp, [], 0), correct_out)

    def test_parse_expr_or_scalar_func_paren_imbalanced_parens(self):
        for testcase in [
            f'((2)',
            f')2(',
            f'))'
        ]:
            with self.subTest(testcase=testcase):
                with self.assertRaises(RemesParserException):
                    parse_expr_or_scalar_func(tokenize(testcase), [], 0)

    #############
    ## handling binops
    #############
    def test_parse_expr_or_scalar_func_simple_binops(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize('@ + 3'), expr([1,2]), 0), (expr([4,5]), 3))

    def test_parse_expr_or_scalar_func_binop_uminus_first_arg(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize('-2*3'), [], 0), (int_node(-6), 4))

    def test_parse_expr_or_scalar_func_uminus_second_arg(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize('2*-3'), [], 0), (int_node(-6), 4))

    def test_parse_expr_or_scalar_func_nested_binops(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize("4 - -2**2 - 3/4"), [], 0), (num(7.25), 10))

    def test_parse_expr_or_scalar_func_parens_first_arg(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize("(3 - 2)*3"), [], 0), (int_node(3), 7))

    def test_parse_expr_or_scalar_func_parens_second_arg(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize("3/(2-4)"), [], 0), (num(-1.5), 7))

    def test_parse_expr_or_scalar_func_parens_not_first_or_last(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize("2**-(1--2)-5"), [], 0), (num(2**-(1--2)-5), 11))


    ###############
    ## apply_indexer_list
    ###############
    def test_apply_indexer_list_one_varname_list(self):
        obj = {'a': 1, 'b': 2, 'c': 3}
        idxrs = [tokenize("[a,b]")]
        self.assertEqual(apply_indexer_list(obj, idxrs), {'a': 1, 'b': 2})

    def test_apply_indexer_list_one_slicer_list(self):
        obj = [1,2,3,4]
        idxrs = [tokenize("[0,3]")]
        self.assertEqual(apply_indexer_list(obj, idxrs), [1,4])

    def test_apply_indexer_list_slicer_then_varname(self):
        obj = [{'a':1, 'b': 2, 'c': 3}, {'a': 4, 'b': 5, 'c': 6}]
        idxrs = [tokenize("[1]"), tokenize(".a")]
        self.assertEqual(apply_indexer_list(obj, idxrs), 4)

    def test_apply_indexer_list_three_slicer_lists(self):
        obj = [[[ 0,  1,  2],
                [ 3,  4,  5]],

               [[ 6,  7,  8],
                [ 9, 10, 11]],

               [[12, 13, 14],
                [15, 16, 17]],

               [[18, 19, 20],
                [21, 22, 23]]]
        idxrs = [tokenize(ix) for ix in ["[0,1,3]", "[1:]", "[0, 1]"]]
        correct_out = [
            [[3,   4]],
            [[9,  10]],
            [[21, 22]]
        ]
        self.assertEqual(apply_indexer_list(obj, idxrs), correct_out)

    def test_apply_indexer_list_two_varname_lists(self):
        obj = {'a': {'b': 1, 'c': 2, 'd': 3}, 'b': {'b': 1, 'c': 2, 'd': 3}, 'e': 'foobar'}
        idxrs = [tokenize("[a,b]"), ['.', 'd']]
        correct_out = {'a': 3, 'b': 3}
        self.assertEqual(apply_indexer_list(obj, idxrs), correct_out)

    def test_apply_indexer_list_slicers_varnames_slicers(self):
        obj = [{'name': 'foo',
  'players': [{'name': 'alice', 'hits': [3, 4, 2, 5]}]},
 {'name': 'bar',
  'players': [{'name': 'carol', 'hits': [7, 3, 0, 5]},
   {'name': 'dave', 'hits': [1, 0, 4, 10]}]}]
        idxrs = [tokenize(ix) for ix in ["[:]", ".players", "[1]"]]
        correct_out = [{'name': 'dave', 'hits': [1, 0, 4, 10]}]
        self.assertEqual(apply_indexer_list(obj, idxrs), correct_out)

    ###############
    ## indexing in parse_expr_or_scalar_func
    ###############
    def test_parse_expr_or_scalar_index_obj(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize('@.foo'), expr({'foo': 'bar'}), 0), (quoted_string('bar'), 3))

    def test_parse_expr_or_scalar_index_arr(self):
        self.assertEqual(parse_expr_or_scalar_func(tokenize('@[0]'), expr([1,2]), 0), (int_node(1), 4))

    def test_parse_expr_or_scalar_index_arr_obj(self):
        obj = expr([{'foo': 1, 'bar': 2},{'foo': 'a', 'bar': 'b'}])
        correct_out = (int_node(1), 6)
        self.assertEqual(parse_expr_or_scalar_func(tokenize('@[0].foo'), obj, 0), correct_out)

    def test_parse_expr_or_scalar_index_obj_arr(self):
        obj = expr({'foo': [1,'a'], 'bar': [2, 'b']})
        correct_out = (expr({'foo': ['a'], 'bar': ['b']}), 10)
        self.assertEqual(parse_expr_or_scalar_func(tokenize('@[foo,bar][1:]'), obj, 0), correct_out)
        
        
    ################
    ## search tests
    ################
    def test_search_s_slice(self):
        self.assertEqual(search('s_slice(`abc`, ::2)', []), 'ac')
        
    def test_search_bool_idx(self):
        x = [{'name': 'a', 'x': 1}, {'name': 'fo', 'x': 2}, {'name': 'fb', 'x': 3}]
        self.assertEqual(search("@[:][s_slice(@.name, 0) == `f`].x", x),
                        [2, 3])


if __name__ == '__main__':
    unittest.main()
