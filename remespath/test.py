from remespath.parser import *
import unittest

def test_parse_indexer(tester, x, out):
    tester.assertEqual((out, len(x)), parse_indexer(x, [], 0))

class RemesPathTester(unittest.TestCase):
    def test_parse_indexer_one_string(self):
        test_parse_indexer(self, 'foo', ['foo'])
        
    def test_parse_indexer_one_string_endsqbk(self):
        test_parse_indexer(self, 'foo]', ['foo'])
        
    def test_parse_indexer_one_int(self):
        test_parse_indexer(self,  '1]', [1])
        
    def test_parse_indexer_one_float(self):
        test_parse_indexer(self, '1.5e3]', [1.5e3])
        
if __name__ == '__main__':
    unittest.main()