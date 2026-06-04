"""Functions used to save and load data."""

import json
import os
import sqlite3

import numpy as np

from src.utils.utils import to_tuple

FUNC_PREFIX = '$'
sql_file = '../../utils/data.sql'
db_name = '/data.db'

#
# Kwargs
#

def save_kwargs(**kwargs):
    """Save kwargs as JSON"""
    def func_to_string(obj):
        """Recursively replace functions with its name preceded by the FUNC_PREFIX"""
        if type(obj) == dict:
            obj = obj.copy()
            for key in obj:
                obj[key] = func_to_string(obj[key])
            return obj
        elif type(obj) == list:
            obj = obj.copy()
            for i, item in enumerate(obj):
                obj[i] = func_to_string(item)
            return obj
        elif type(obj) == np.ndarray:
            return to_tuple(obj)
        elif hasattr(obj, '__name__'):
            return f'{FUNC_PREFIX}.{obj.__name__}'
        else:
            return obj
    print(f'Saving kwargs to {kwargs['saves_path']}/kwargs.json')
    os.makedirs(kwargs['saves_path'], exist_ok=True)
    with open(kwargs['saves_path'] + '/kwargs.json', 'w') as f:
        json.dump(func_to_string(kwargs.copy()), f, indent=4)


def load_kwargs(saves_path):
    """Load kwargs from JSON"""
    def string_to_func(obj):
        """Recursively replace strings preceded by the FUNC_PREFIX with the imported function of the same name"""
        if type(obj) is dict:
            for key in obj:
                obj[key] = string_to_func(obj[key])
        elif type(obj) is list:
            for i, item in enumerate(obj):
                obj[i] = string_to_func(item)
        elif type(obj) is type('') and obj.startswith(FUNC_PREFIX + '.'):
            # module = __import__(GP_FILE)
            import src.models as module
            return getattr(module, obj[len(FUNC_PREFIX) + 1:])
        return obj
    print(f'Loading kwargs from {saves_path}/kwargs.json')
    with open(saves_path+'/kwargs.json', 'rb') as f:
        kwargs = string_to_func(json.load(f))
    kwargs['saves_path'] = saves_path
    return kwargs

#
# Database
#

def create_db(**kwargs):
    # Each test is saved in its own directory which is passed through the path
    print(f'Creating database {kwargs['saves_path']+db_name}')
    os.makedirs(kwargs['saves_path'], exist_ok=True)
    con = sqlite3.connect(kwargs['saves_path']+db_name)
    # try:
    with open(sql_file, 'r') as f:
        init_sql = f.read()
        with con:
            cur = con.cursor()
            cur.executescript(init_sql)
            # con.commit() # Not needed
    # except:
    #     print('ERROR')
    con.close()


def update_db(pops, fits, generation, **kwargs):
    """Insert values from the pop and fits to the database"""
    print(f'Updating Database {kwargs['saves_path']+db_name}')

    # Format data into a single list
    gen_start = generation - len(pops) + 1
    data = [
        [
            kwargs['test_name'],
            kwargs['seed'],
            gen_start + gen_offset,
            ind,
            fits[gen_offset][ind],
            str(pops[gen_offset][ind]),
        ]
        for gen_offset in range(len(pops))
        for ind in range(len(pops[gen_offset]))
    ]

    con = sqlite3.connect(kwargs['saves_path'] + db_name, timeout=kwargs['update_timeout'])
    with con:
        cur = con.cursor()
        cur.executemany("INSERT INTO data VALUES(?, ?, ?, ?, ?, ?)", data)
        # con.commit()
    con.close()


def sql_query(query, **kwargs):
    con = sqlite3.connect(kwargs['path'] + db_name)
    cur = con.cursor()
    res = list(cur.execute(query))
    # for i in res:
    #     print(i)
    con.close()
    return res



if __name__ == '__main__':
    # name = 'unstable_self_rep_0'
    # fits = load_fits(**kwargs)
    #
    # pop = load_pop(0, 0, **kwargs)


    # Run query
    # q = 'select gen, count(gen) from data group by gen'
    # q = 'SELECT test, seed, AVG(fit) FROM data GROUP BY test, seed'
    # q = """
    # SELECT test, MIN(fit)
    # FROM (
    #     SELECT test, seed, AVG(fit) AS fit
    #     FROM data
    #     GROUP BY test, seed
    #     )
    # GROUP BY test
    # """
    # q = """
    # SELECT test, gen, MIN(fit)
    # FROM (
    #     SELECT test, seed, gen, AVG(fit) AS fit
    #     FROM data
    #     GROUP BY test, seed, gen
    #     )
    # GROUP BY test, gen
    # """
    q = """
    SELECT *
    FROM data
    WHERE fit = 0
    GROUP BY data
    """
    name = 'test/node'
    kwargs = load_kwargs(name, '../../saves/')
    kwargs['path'] = '../../saves/' + name + '/'

    for i in sql_query(q, **kwargs):
        print(i)



    # Interactive SQL terminal
    # con = sqlite3.connect(kwargs['path'] + db_name)
    # cur = con.cursor()
    # try:
    #     while True:
    #         i = input()
    #         if i == '':
    #             break
    #         res = cur.execute(i)
    #         for i in res:
    #             print(i)
    #         # print(res.fetchall())
    # finally:
    #     con.close()
    #     print('EXITING')


