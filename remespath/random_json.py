import json
import string
import random

def random_simple_arr():
    out = []
    for ii in range(2):
        out.append(random.randint(0, 1e10))
    for ii in range(2):
        out.append(random.random())
    for ii in range(2):
        out.append(''.join(random.choices(string.printable, k=5)))
        
    return out
    
    
def random_simple_dict():
    return dict(zip('abcdef', random_simple_arr()))


def random_fancy_dict(nkeys, composition=None):
    if composition is None:
        composition = [1/3, 1/2, 2/3, 5/6]
    int_comp, float_comp, str_comp, obj_comp = [int(nkeys*x) for x in composition]
    keys = string.ascii_letters[:nkeys]
    int_keys = keys[:int_comp]
    float_keys = keys[int_comp:float_comp]
    str_keys = keys[float_comp:str_comp]
    obj_keys = keys[str_comp:obj_comp]
    arr_keys = keys[obj_comp:]
    out = {}
    for k in int_keys:
        out[k] = random.randint(0, 1e10)
    for k in float_keys:
        out[k] = random.random()
    for k in str_keys:
        out[k] = ''.join(random.choices(string.printable, k=10))
    for k in obj_keys:
        out[k] = random_simple_dict()
    for k in arr_keys:
        out[k] = random_simple_arr()
        
    return out
    
    
def random_fancy_json(nrows, nkeys, dict_composition=None):
    return [random_fancy_dict(nkeys, dict_composition) for ii in range(nrows)]
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('nrows', nargs='?', type=int, default=10, help='number of rows in fancy random json')
    parser.add_argument('nkeys', nargs='?', type=int, default=10, help='number of keys per dict in fancy random json')
    parser.add_argument('-d', '--dict_only', help='only get a random dict', action='store_true')
    parser.add_argument('-c', '--composition', nargs='+', type=float, help='fraction of keys for ints, floats, strings, and dicts, in that order. All must be numbers less than 1', action='store', default=None)
    args = parser.parse_args()
    if args.dict_only:
        out = random_fancy_dict(args.nkeys, args.composition)
    else:
        out = random_fancy_json(args.nrows, args.nkeys, args.composition)
        
    print(json.dumps(out))