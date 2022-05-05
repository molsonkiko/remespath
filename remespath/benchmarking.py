import cProfile
import json
import os
from remespath.parser import search
remes_dir = os.path.dirname(__file__)

with open(os.path.join(remes_dir, 'data', 'baseball.json')) as f:
    baseball = json.load(f)
    
with open(os.path.join(remes_dir, 'data', 'big_random.json')) as f:
    big_random = json.load(f)
    
cProfile.run("for ii in range(2): rantext = search('@[:].z =~ `(?i)[a-z]{5}`', big_random)", 
    os.path.join(remes_dir, "cProfile results simple query large file.txt"))

cProfile.run('''for ii in range(200):
    search("sort_by(@[:]"
            "{name: @.name, hits: "
                "sum(flatten(@.players[:].hits[@ > 0]))"
            "}, "
            "hits, true)",
        baseball)''',
    os.path.join(remes_dir, "cProfile results complex query small file.txt"))