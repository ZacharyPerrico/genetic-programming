"""Functions used to save and load data."""
import importlib
import copy
import json
import os
import sqlite3

import numpy as np

from src.utils.utils import to_tuple

FUNC_PREFIX = '$'
sql_file = '../../utils/data.sql'
db_name = '/data.db'

#
# Kwarg Generators
#

def generate_reps(**kwargs):
    """Yields kwargs with unique seeds and rngs for each replicate"""
    for _ in range(kwargs['num_reps']):
        # Assign seed and RNG
        kwargs['seed'] = (np.random.randint(0, 2**64, dtype='uint64'))
        kwargs['rng'] = np.random.default_rng(kwargs['seed'])
        yield kwargs.copy()


def generate_tests(test_keys, test_values, **kwargs):
    """Convert simulation kwargs containing test_kwargs into a list of all the kwargs"""
    kwargs['num_tests'] = len(test_values)
    for test_num in range(kwargs['num_tests']):

        # Update with test-specific values
        rep_kwargs = copy.deepcopy(kwargs)
        for key, value in zip(test_keys, test_values[test_num]):
            rep_kwargs[key] = value

        # Add no-operation as a possible recombination
        prob_noop = 1 - sum(rep_kwargs['recombination_probs'])
        if prob_noop > 0:
            rep_kwargs['recombination_funcs'].append(None)
            rep_kwargs['recombination_probs'].append(prob_noop)

        # Add no-operation as a possible mutation
        prob_noop = 1 - sum(rep_kwargs['mutation_probs'])
        if prob_noop > 0:
            rep_kwargs['mutation_funcs'].append(None)
            rep_kwargs['mutation_probs'].append(prob_noop)

        yield rep_kwargs


#
# Format Kwargs
#

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
        return f'{FUNC_PREFIX}{obj.__module__}.{obj.__name__}'
    else:
        return obj


def string_to_func(obj):
    """Recursively replace strings preceded by the FUNC_PREFIX with the imported function of the same name"""
    if type(obj) is dict:
        for key in obj:
            obj[key] = string_to_func(obj[key])
    elif type(obj) is list:
        for i, item in enumerate(obj):
            obj[i] = string_to_func(item)
    elif type(obj) is type('') and obj.startswith(FUNC_PREFIX):
        module, name = obj[len(FUNC_PREFIX):].rsplit('.', 1)
        module = importlib.import_module(module)
        return getattr(module, name)
    return obj


#
# Save and Load Kwargs
#

def save_kwargs(**kwargs):
    """Save kwargs as JSON"""
    print(f'Saving kwargs to {kwargs['saves_path']}/kwargs.json')
    os.makedirs(kwargs['saves_path'], exist_ok=True)
    json_data = func_to_string(kwargs.copy())
    with open(kwargs['saves_path'] + '/kwargs.json', 'w') as f:
        json.dump(json_data, f, indent=4)


def load_kwargs(saves_path):
    """Load kwargs from JSON"""
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
            kwargs['test'],
            kwargs['seed'],
            gen_start + gen_offset,
            ind,
            fits[gen_offset][ind],
            kwargs['save_formater_func']((pops[gen_offset][ind])),
        ]
        for gen_offset in range(len(pops))
        for ind in range(len(pops[gen_offset]))
    ]

    con = sqlite3.connect(kwargs['saves_path']+db_name, timeout=kwargs['update_timeout'])
    with con:
        cur = con.cursor()
        cur.executemany("INSERT INTO data VALUES(?, ?, ?, ?, ?, ?)", data)
        # con.commit()
    con.close()


def sql_query(query, return_col_names=False, **kwargs):
    con = sqlite3.connect(kwargs['saves_path']+db_name)
    cur = con.cursor()
    res = cur.execute(query)
    if return_col_names:
        col_names = [desc[0] for desc in res.description]
    res = list(res)
    con.close()
    if return_col_names:
        res = res, col_names
    return res



def create_kwarg_table(**test_kwargs):
    """Use an instance of a test's kwargs to create a table to store the kwargs for each test"""
    cols = ''
    for key in test_kwargs:
        value = test_kwargs[key]
        if type(value) == str:
            datatype = 'TEXT'
        elif type(value) == int:
            datatype = 'INT'
        else:
            datatype = 'ANY'
        cols += f'{key} {datatype},'
    cols += 'PRIMARY KEY (test)'
    sql_query(f"""CREATE TABLE IF NOT EXISTS kwargs ({cols})""", **test_kwargs)


def update_kwarg_table(**test_kwargs):
    """Save the kwargs for each test"""
    num_cols = len(test_kwargs)
    k = func_to_string(test_kwargs)
    # k = kwargs
    # Reorder kwargs
    # k = {'test': kwargs['test']} | kwargs
    values = list(k.values())
    for i, value in enumerate(values):
        new_value = str(value)
        values[i] = new_value
    con = sqlite3.connect(test_kwargs['saves_path'] + db_name, timeout=test_kwargs['update_timeout'])
    with con:
        cur = con.cursor()
        cur.execute(f"INSERT INTO kwargs VALUES({','.join('?' * num_cols)})", values)
    con.close()




# if __name__ == '__main__':
    #
    # # Run query
    # name = 'daggp/test'
    # kwargs = load_kwargs('../../saves/'+name)
    #
    # test_kwargs = generate_tests(**kwargs)
    #
    #
    #
    # for test_kwarg in test_kwargs:
    #     update_kwarg_table(**test_kwarg)
    #
    # update_kwarg_table(**kwargs)
    #
    #
    #
    #
    # quit()

    # q = """
    #     WITH sub AS (
    #         SELECT test, COUNT() AS c
    #         FROM data
    #         WHERE fit = 0 AND gen = 299
    #         GROUP BY test
    #     )
    #     SELECT *
    #     FROM sub
    # """
    # q = 'SELECT gen, count(gen) FROM data GROUP BY gen'
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
    # q = """
    # SELECT *
    # FROM data
    # WHERE fit = 0
    # GROUP BY data
    # """


    #
    # for i in sql_query(q, **kwargs):
    #     print(i)
    #


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


