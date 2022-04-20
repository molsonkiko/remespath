# AST nodes have this structure:
# {"type": <node type>", children: [], "value": ""}
LANGUAGE_SPEC = '''
expr ::= json (ws (indexer_list | projection))*
projection ::= object_projection | array_projection
object_projection ::= "{" ws key_value_pair ws ("," ws key_value_pair ws)* "}"
array_projection ::= "{" ws expr_function ws ("," ws expr_function ws)* "}"
key_value_pair ::= string ws ":" ws expr_function
json ::= cur_json_func | json_string
cur_json_func ::= "@"
indexer_list ::= indexer+
indexer ::= "." varname
            | "[" ws boolean_index ws "]"
            | "[" ws varname_list ws "]"
            | "[" ws slicer_list ws "]"
expr_function ::= expr
                  | (expr | scalar_function) ws binop ws expr
                  | expr ws binop ws (expr | scalar_function)
                  | expr_arg_function
                  | "(" expr_function ")"
scalar_function ::= scalar
                    | scalar_function ws binop ws scalar_function
                    | scalar_arg_function
                    | "(" scalar_function ")"
boolean_index ::= expr_function
expr_arg_function ::= unquoted_string ws "(" ws expr_function ws ("," ws (expr_function | scalar_function) ws)* ")"
scalar_arg_function ::= unquoted_string ws "(" ws scalar_function ws ("," ws scalar_function ws)* ")"
slicer_list ::= slicer ws ("," ws slicer ws)*
varname_list ::= varname ws ("," ws varname ws)*
slicer ::= int | int? ":" int? ":" int?
scalar ::= quoted_string | num | regex | constant
varname ::= string | regex
string ::= quoted_string | unquoted_string
regex ::= "g" quoted_string
json_string ::= "j" quoted_string
quoted_string ::= "`" ascii_char* "`" ; "`" inside the string must be escaped by "\\"; see the BACKTICK_STRING_REGEX below
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
          | "/" | "//" | "*{1, 2}" | "in"
'''
import re
inf = float('inf')


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


def cur_json_func(func=None):
    '''a function of the user-supplied json. Undefined at compile time.
If no function supplied, defaults to the identity function.'''
    # func = func or identity
    return {'type': 'cur_json_func', 'value': func}


def delim(value):
    '''any one of ,()[]{}:.'''
    return {'type': 'delim', 'value': value}


INDEXER_SUBTYPES = {'slicer_list', 'varname_list', 'boolean_index'}
def indexer_list(*indexers):
    return {'type': 'indexer_list', 'children': indexers}


def int_node(value):
    return {'type': 'int', 'value': value}


EXPR_SUBTYPES = {'expr', 'cur_json_func'}
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
def scalar(value):
    return {'type': 'scalar', 'value': value}


SLICER_SUBTYPES = {'int', 'slicer'}
def slicer(ints):
    return {'type': 'slicer', 'value': slice(*ints)}


def slicer_list(slicers):
    return {'type': 'slicer_list', 'children': slicers}


def string(value):
    return {'type': 'string', 'value': value}


VARNAME_SUBTYPES = {'string', 'regex'}

def varname_list(nodes):
    return {'type': 'varname_list', 'children': nodes}


AST_TOK_BUILDER_MAP = {
    # the ast function that makes each type of token
    'binop': binop_function,
    'bool': bool_node,
    'cur_json_func': cur_json_func,
    'expr': expr,
    'int': int_node,
    'null': null_node,
    'num': num,
    'projection': projection,
    'regex': regex,
    'scalar': scalar,
    'slicer': slicer,
    'slicer_list': slicer_list,
    'string': string,
    'varname_list': varname_list,
}

AST_TYPE_BUILDER_MAP = {
    bool: bool_node,
    dict: expr,
    float: num,
    function_type: cur_json_func,
    int: int_node,
    list: expr,
    re.Pattern: regex,
    str: string,
}