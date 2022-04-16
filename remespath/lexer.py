import re
import unittest

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


def tokenize(query):
    toks = [tok.groups()[0] for tok in MASTER_REGEX.finditer(query)]
    # for open_paren, close_paren in [('(', ')'), ('[', ']'), ('{', '}')]:
        # open_ct = 0
        # first_open = None
        # for ii, t in enumerate(toks):
            # if t == open_paren:
                # open_ct += 1
                # if first_open is None:
                    # first_open = ii
            # elif t == close_paren:
                # open_ct -= 1
                # if open_ct < 0:
                    # raise RemesLexerException(f"Unmatched '{close_paren}' at token {ii}\n{toks}")
                # if open_ct == 0:
                    # # every time parity is reached again, reset the count
                    # first_open = None
        # if open_ct > 0:
            # raise RemesLexerException(f"Unmatched '{open_paren}' at token {first_open}\n{toks}")
            
    return toks


class RemesPathLexerTester(unittest.TestCase):
    def test_tokenize(self):
        for query_example, correct_out in [
            ('@.foo[bar,baz][:][1:3]', [
            '@', '.', 'foo', '[', 'bar', ',', 'baz', ']', '[', ':', ']',
            '[', '1', ':', '3', ']']),
            ('[foo == g`x+12` | sum(2.0**`a`)]', ['[', 'foo', '==', 'g`x+12`', '|', 'sum', '(', '2.0', '**', '`a`', ')', ']']),
            ('@{`a`: 2e+2*((3e-1))}', ['@', '{', '`a`', ':',  '2e+2', '*', '(', '(', '3e-1', ')', ')', '}']),
            ('{`x`: 3.5e2, `y\\``: -j`[1]`}', ['{', '`x`', ':', '3.5e2', ',', '`y\\``', ':', '-', 'j`[1]`', '}']),
            ('2 < 3 > 4 <= 5 >= 6', ['2', '<', '3', '>', '4', '<=', '5', '>=', '6']),
            ('`abc` =~ g`\\`` & j``', ['`abc`', '=~', 'g`\\``', '&', 'j``']),
            ('3 // 6/-2 != true', ['3', '//', '6', '/', '-', '2', '!=', 'true']),
            (' a % b9 ^ _c ', ['a', '%', 'b9', '^', '_c']),
        ]:
            # print(get_tokens(query_example))
            self.assertEqual(correct_out, list(tokenize(query_example)))


if __name__ == '__main__':
    unittest.main()