# AST nodes have this structure:
# {"type": <node type>", children: [], "value": ""}
LANGUAGE_SPEC = '''
expr ::= json (ws (indexer_list | projection))*
projection ::= object_projection | array_projection
object_projection ::= "{" ws key_value_pair ws ("," ws key_value_pair ws)* "}"
array_projection ::= "{" ws expr_function ws ("," ws expr_function ws)* "}"
key_value_pair ::= quoted_string ws ":" ws expr_function
json ::= current_document | json_string
current_document ::= "@"
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
varname ::= unquoted_string | quoted_string | regex
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


def array_projection(children):
    return {'type': 'array_projection', 'children': children}
    

def binop_function(func, precedence, first=None, second=None):
    return {'type': 'binop', 'children': [first, second], 'value': [func, precedence]}


def bool_node(value):
    return {'type': 'bool', 'value': value}
    

def boolean_index(value):
    return {'type': 'boolean_index', 'children': value}


# def arg_function(argfunc, subtype):
    # if argfunc.max_args == inf:
        # args = []
    # else:
        # args = [None for ii in range(argfunc.max_args)]
    # return {'type': f'{subtype}_arg_function', 
            # 'children': args, 
            # 'value': argfunc}


INDEXER_SUBTYPES = {'slicer_list', 'varname_list', 'boolean_index'}
def indexer_list(*indexers):
    return {'type': 'indexer_list', 'children': indexers}


def int_node(value):
    return {'type': 'int', 'value': value}


def expr(json_):
    return {'type': 'expr', 'value': json_}


def null_node():
    return {'type': 'null', 'value': None}


def num(value):
    return {'type': 'num', 'value': value}


def object_projection(children):
    return {'type': 'object_projection', 'children': children}


def quoted_string(value):
    return {'type': 'quoted_string', 'value': value}


def regex(value):
    return {'type': 'regex', 'value': value}


SCALAR_SUBTYPES = {'int', 'num', 'bool', 'quoted_string', 'unquoted_string', 'regex', 'null'}
ALL_BASE_SUBTYPES = {'expr'} | SCALAR_SUBTYPES
def scalar(value):
    return {'type': 'scalar', 'value': value}
    

SLICER_SUBTYPES = {'int', 'slicer'}
def slicer(ints):
    return {'type': 'slicer', 'value': slice(*ints)}


def slicer_list(slicers):
    return {'type': 'slicer_list', 'children': slicers}


def unquoted_string(value):
    return {'type': 'unquoted_string', 'value': value}


VARNAME_SUBTYPES = {'unquoted_string', 'quoted_string', 'regex'}
# def varname(node):
    # return {"type": "varname", "value": node['value']}

def varname_list(nodes):
    # assert all(node['type'] == 'varname' for node in nodes) 
    return {'type': 'varname_list', 'children': nodes}
    
    
AST_TOK_BUILDER_MAP = {
    # the ast function that makes each type of token
    'array_projection': array_projection,
    'binop': binop_function,
    'bool': bool_node,
    'expr': expr,
    # 'expr_arg_function': arg_function,
    'int': int_node,
    'null': null_node,
    'num': num,
    'object_projection': object_projection,
    'quoted_string': quoted_string,
    'regex': regex,
    'scalar': scalar,
    # 'scalar_arg_function': arg_function,
    'slicer': slicer,
    'slicer_list': slicer_list,
    'unquoted_string': unquoted_string,
    'varname_list': varname_list,
}

AST_TYPE_BUILDER_MAP = {
    bool: bool_node,
    dict: expr,
    float: num,
    int: int_node,
    list: expr,
    re.Pattern: regex,
    str: quoted_string,
}