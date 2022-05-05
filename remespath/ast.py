# AST nodes have this structure:
# {"type": <node type>", children: [], "value": ""}
LANGUAGE_SPEC = '''
expr ::= json (ws indexer_list)? projection?
projection ::= object_projection | array_projection
object_projection ::= l_curlybrace ws key_value_pair ws (comma ws key_value_pair ws)* r_curlybrace
array_projection ::= l_curlybrace ws expr_function ws (comma ws expr_function ws)* r_curlybrace
l_curlybrace ::= "{"
r_curlybrace ::= "}"
key_value_pair ::= string ws colon ws expr_function
json ::= cur_json | json_string
cur_json ::= "@"
indexer_list ::= double_dot_indexer? indexer+
indexer ::= dot single_varname
            | l_squarebrace ws boolean_index ws r_squarebrace
            | l_squarebrace ws varname_list ws r_squarebrace
            | l_squarebrace ws slicer_list ws r_squarebrace
double_dot_indexer ::= dot single_varname
                        | dot dot l_squarebrace ws boolean_index ws r_squarebrace
                        | dot dot l_squarebrace ws varname_list ws r_squarebrace
                        | dot dot l_squarebrace ws slicer_list ws r_squarebrace
l_squarebrace ::= "["
r_squarebrace ::= "]"
expr_function ::= expr
                  | (expr | scalar_function) ws binop ws expr
                  | expr ws binop ws (expr | scalar_function)
                  | expr_arg_function
                  | lparen expr_function rparen
scalar_function ::= scalar
                    | scalar_function ws binop ws scalar_function
                    | scalar_arg_function
                    | lparen scalar_function rparen
boolean_index ::= expr_function
expr_arg_function ::= unquoted_string ws lparen ws expr_function ws (comma ws (expr_function | scalar_function) ws)* rparen
scalar_arg_function ::= unquoted_string ws lparen ws scalar_function ws (comma ws scalar_function ws)* rparen
single_varname ::= varname | star
slicer_list ::= star | slicer ws (comma ws slicer ws)*
star ::= "*"
varname_list ::= varname ws (comma ws varname ws)*
slicer ::= int | int? colon int? colon int?
scalar ::= quoted_string | num | regex | constant
varname ::= string | regex
string ::= quoted_string | unquoted_string
regex ::= g quoted_string
json_string ::= j quoted_string
quoted_string ::= backtick ascii_char* backtick ; "`" inside the string must be escaped by "\\"; see the BACKTICK_STRING_REGEX below
unquoted_string ::= "[a-zA-Z_][a-zA-Z_0-9]*"
ascii_char ::= "[\x00-\xff]"
num ::= int dec_part? exp_part?
int ::= "(-?(?:0|[1-9]\d*))"
dec_part ::= "(\.\d+)"
exp_part ::= "([eE][-+]?\d+)?"
constant ::= bool | "Infinity" | null | "NaN" | "-Infinity"
null ::= "null"
bool ::= "true" | "false"
ws ::= "[ \t\n\r]*"
binop ::= "&" | "|" | "^" | "=~" | "[=><!]=" | "<" | ">" | "+" | "-"
          | "/" | "//" | star | star star | "in"
colon ::= ":"
comma ::= ","
dot ::= "."
g ::= "g"
j ::= "j"
'''
import re
inf = float('inf')

def identity(x): return x


def arg_function(argfunc):
    if argfunc.max_args == inf:
        args = []
    else:
        args = [None for ii in range(argfunc.max_args)]
    return {'type': 'arg_function',
            'children': args,
            'value': argfunc}


function_type = type(arg_function)


def binop_function(func, precedence, first=None, second=None):
    return {'type': 'binop', 'children': [first, second], 'value': [func, precedence]}


def bool_node(value):
    return {'type': 'bool', 'value': value}


def cur_json(func=None):
    '''a function of the user-supplied json. Undefined at compile time.
If no function supplied, defaults to the identity function.'''
    func = func or identity
    return {'type': 'cur_json', 'value': func}


def delim(value):
    '''any one of ,()[]{}:.'''
    return {'type': 'delim', 'value': value}


INDEXER_SUBTYPES = {'slicer_list', 'varname_list', 'boolean_index', 'star_indexer_list'}


def int_node(value):
    return {'type': 'int', 'value': value}


EXPR_SUBTYPES = {'expr', 'cur_json', 'projection'}
def expr(json_):
    return {'type': 'expr', 'value': json_}


def null_node():
    return {'type': 'null', 'value': None}


def num(value):
    return {'type': 'num', 'value': value}


def projection(value):
    return {'type': 'projection', 'value': value}


def regex(value):
    return {'type': 'regex', 'value': value}


SCALAR_SUBTYPES = {'int', 'num', 'bool', 'string', 'regex', 'null'}
ALL_BASE_SUBTYPES = EXPR_SUBTYPES | SCALAR_SUBTYPES
# def scalar(value):
    # return {'type': 'scalar', 'value': value}


SLICER_SUBTYPES = {'int', 'slicer'}
def slicer(ints):
    return {'type': 'slicer', 'value': slice(*ints)}


def slicer_list(slicers):
    return {'type': 'slicer_list', 'children': slicers}


def star_indexer_list():
    return {'type': 'star_indexer_list', 'children': None}


def string(value):
    return {'type': 'string', 'value': value}


VARNAME_SUBTYPES = {'string', 'regex'}

def varname_list(nodes):
    return {'type': 'varname_list', 'children': nodes}


AST_TOK_BUILDER_MAP = {
    # the ast function that makes each type of token
    'binop': binop_function,
    'bool': bool_node,
    'cur_json': cur_json,
    'expr': expr,
    'int': int_node,
    'null': null_node,
    'num': num,
    'projection': projection,
    'regex': regex,
    # 'scalar': scalar,
    'slicer': slicer,
    'slicer_list': slicer_list,
    'string': string,
    'varname_list': varname_list,
    'star_indexer_list': star_indexer_list,
}

AST_TYPE_BUILDER_MAP = {
    bool: bool_node,
    dict: expr,
    float: num,
    function_type: cur_json,
    int: int_node,
    list: expr,
    re.Pattern: regex,
    str: string,
}