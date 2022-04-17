import json
import re
import unittest
from remespath.ast import *
from remespath.functions import BINOPS, CONSTANTS, FUNCTIONS

class RemesLexerException(Exception):
    pass


MASTER_REGEX = re.compile(
    r'(&|\||\^|\+|-|/{1,2}|\*{1,2}|%|' # most arithmetic and bitwise operators
    r'[=!]=|[><]=?|=~|' # comparision operators
    r'[gj]?`(?:[^`]|(?<=\\)`)*(?<!\\)`|' # backtick string with optional g or j prefix
    r'\[|\]|\(|\)|\{|\}' # close and open parens, squarebraces, and curlybraces
    '|(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][-+]?\\d+)?' # numbers
    '|,|:|\.|@|' # commas, colons, periods, '@' symbol
    '[a-zA-Z_][a-zA-Z_0-9]*)' # unquoted_string
)

num_match = re.compile(("(-?(?:0|[1-9]\d*))"   # any int or 0
                "(\.\d+)?" # optional decimal point and digits after
                "([eE][-+]?\d+)?" # optional scientific notation
                )).match
NUM_START_CHARS = set('0123456789')
DELIMITERS = set(',[](){}.:')


def tokenize(query):
    out = []
    for mtch in MASTER_REGEX.finditer(query):
        t = mtch.groups()[0]
        c = t[0]
        if c == '@':
            out.append(current_json())
        elif c in DELIMITERS:
            out.append(delim(c))
        elif c == '`':
            out.append(string(t[1:-1].replace('\\`', '`')))
        elif c in NUM_START_CHARS:
            try:
                g1, g2, g3 = num_match(t).groups()
                if g2:
                    if g3:
                        out.append(num(float(g1+g2+g3)))
                    else:
                        out.append(num(float(g1+g2)))
                elif g3:
                    out.append(num(float(g1+g3)))
                else:
                    out.append(int_node(int(g1)))
            except:
                raise RemesLexerException(f"Tokens beginning with digits ({t}) must be numbers.")
            # try:
                # # this is the alternative - it performs better for ints but
                # # worse for floats
                # n = int(t)
                # out.append(int_node(n))
            # except:
                # out.append(num(float(t)))
                # except:
                    # raise RemesLexerException("Tokens beginning with digits must be numbers.")
        elif c == 'j':
            if t[1:2] == '`':
                try:
                    out.append(expr(json.loads(t[2:-1].replace('\\`', '`'))))
                except json.JSONDecodeError as ex:
                    raise RemesLexerException(f"Malformed json ({ex})")
        elif c == 'g':
            if t[1:2] == '`':
                try:
                    out.append(regex(re.compile(t[2:-1].replace('\\`', '`'))))
                except Exception as ex:
                    raise RemesLexerException(f"Malformed regex ({ex})")
        elif t in BINOPS:
            out.append(binop_function(*BINOPS[t]))
        elif t in FUNCTIONS:
            out.append(arg_function(FUNCTIONS[t]))
        elif t in CONSTANTS:
            const = CONSTANTS[t]
            if isinstance(const, float):
                # inf, -inf, nan
                out.append(num(const))
            elif isinstance(const, bool):
                out.append(bool_node(const))
            else:
                out.append(null_node())
        else:
            out.append(string(t))
            
    return out
        


class RemesPathLexerTester(unittest.TestCase):
    maxDiff = None
    def test_tokenize(self):
        for query_example, correct_out in [
            ('@.foo[b,z][1:3]', [
            current_json(), delim('.'), string('foo'), delim('['), string('b'), delim(','), string('z'), delim(']'), delim('['), int_node(1), delim(':'), int_node(3), delim(']')]),
            ('foo == g`x+12` | sum(2.0**`a`)', [string('foo'), binop_function(*BINOPS['==']), regex(re.compile('x+12')), binop_function(*BINOPS['|']), arg_function(FUNCTIONS['sum']), delim('('), num(2.), binop_function(*BINOPS['**']), string('a'), delim(')')]),
            ('@{`a`: 2e+2*(3.5e-1)}', [current_json(), delim('{'), string('a'), delim(':'), num(2e2), binop_function(*BINOPS['*']), delim('('), num(3.5e-1), delim(')'), delim('}')]),
            ('{`y\\``: -j`[1]`}', [delim('{'), string('y`'), delim(':'), binop_function(*BINOPS['-']), expr([1]), delim('}')]),
            ('< > <=', [binop_function(*BINOPS['<']), binop_function(*BINOPS['>']), binop_function(*BINOPS['<='])]),
            ('`abc` =~ g`\\`` & j`"a"`', [string('abc'), binop_function(*BINOPS['=~']), regex(re.compile('`')), binop_function(*BINOPS['&']), expr('a')]),
            ('null // str(true)', [null_node(), binop_function(*BINOPS['//']), arg_function(FUNCTIONS['str']), delim('('), bool_node(True), delim(')')]),
            ('b9 _c ', [string('b9'), string('_c')]),
        ]:
            with self.subTest(query_example=query_example,correct_out=correct_out):
                # print(get_tokens(query_example))
                self.assertEqual(correct_out, list(tokenize(query_example)))


if __name__ == '__main__':
    unittest.main()