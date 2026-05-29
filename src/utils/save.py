"""Functions used to save and load data."""

import glob
import json
import os
import sqlite3

import numpy as np

from src.utils.utils import to_tuple


FUNC_PREFIX = '$'

def save_kwargs(**kwargs):
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
    kwargs_path = f'{kwargs['saves_path']}{kwargs['name']}/'
    print(f'Saving kwargs to {kwargs_path}kwargs.json')
    os.makedirs(kwargs_path, exist_ok=True)
    with open(kwargs_path + 'kwargs.json', 'w') as f:
        json.dump(func_to_string(kwargs.copy()), f, indent=4)


def load_kwargs(name, saves_path):
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
    path = f'{saves_path}{name}/kwargs.json'
    print(f'Loading kwargs from {path}')
    with open(path, 'rb') as f:
        kwargs = string_to_func(json.load(f))
    kwargs['saves_path'] = saves_path
    return kwargs




def save_run(test_path, pops, fits, **kwargs):
    # Each test is saved in its own directory which is passed through the path
    os.makedirs(test_path, exist_ok=True)
    test_path = f'{test_path}/{kwargs["seed"]}/'
    print(f'Saving run to {test_path}')
    os.makedirs(test_path, exist_ok=True)
    np.save(test_path + 'pops', pops)
    np.save(test_path + 'fits', fits)



def load_runs(**kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    pops = []
    fits = []
    tests = [test[0] for test in kwargs['test_kwargs'][1:]]
    for test in tests:
        pops.append([])
        fits.append([])
        test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test}/*/'
        for run_file_name in glob.glob(test_path):
            print(f'Loading run from {run_file_name}')
            pops[-1].append(np.load(run_file_name+'pops.npy', allow_pickle=True))
            fits[-1].append(np.load(run_file_name+'fits.npy'))
    # pops = np.array(pops, dtype=[('verts','object'),('edges','object')])
    pops = np.array(pops, dtype=object)
    fits = np.array(fits)
    return pops, fits


def load_fits(**kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    fits = []
    test_names = [test[0] for test in kwargs['test_kwargs'][1:]]
    for test_name in test_names:
        fits.append([])
        test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test_name}/*/'
        for run_file_name in sorted(glob.glob(test_path)):
            print(f'Loading fitness from {run_file_name}')
            fits[-1].append(np.load(run_file_name+'fits.npy'))
    fits = np.array(fits)
    return fits


def load_pops(**kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    pops = []
    test_names = [test[0] for test in kwargs['test_kwargs'][1:]]
    for test_name in test_names:
        pops.append([])
        test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test_name}/*/'
        for run_file_name in glob.glob(test_path):
            print(f'Loading run from {run_file_name}')
            pops[-1].append(np.load(run_file_name+'pops.npy', allow_pickle=True))
    # pops = np.array(pops, dtype=[('verts','object'),('edges','object')])
    pops = np.array(pops, dtype=object)
    return pops


def load_pop(test, run, **kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    test_name = kwargs['test_kwargs'][1:][test][0]
    test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test_name}/*/'
    run_file_name = sorted(glob.glob(test_path))[run]
    print(f'Loading population from {run_file_name}')
    pop = np.load(run_file_name+'pops.npy', allow_pickle=True)
    return pop


def load_seed(test, run, **kwargs):
    """Returns the seed used for a run given the test name and the index of the run"""
    test_name = kwargs['test_kwargs'][1:][test][0]
    test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test_name}/*/'
    run_file_name = sorted(glob.glob(test_path))[run]
    run_seed = run_file_name.split('\\')[-2]
    return run_seed


def load_seeds(test, **kwargs):
    """Returns the seed used for a run given the test name and the index of the run"""
    test_name = kwargs['test_kwargs'][1:][test][0]
    test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test_name}/*/'
    run_file_names = sorted(glob.glob(test_path))
    run_seeds = [int(run_file_name.split('\\')[-2]) for run_file_name in run_file_names]
    return run_seeds



# schema = """
# DROP TABLE IF EXISTS data;
#
# CREATE TABLE IF NOT EXISTS data (
#   test TEXT,
#   seed INT NOT NULL,
#   gen INT,
#   id INT,
#   fitness REAL,
#   data TEXT,
#   PRIMARY KEY (test, seed, gen, id)
# );
#
# CREATE TABLE IF NOT EXISTS kwargs (
#     test TEXT,
#     PRIMARY KEY (test)
# )
# """

sql_file = '../../utils/data.sql'
db_name = 'data.db'

def create_db(**kwargs):
    # Each test is saved in its own directory which is passed through the path
    os.makedirs(kwargs['path'], exist_ok=True)
    print(f'Creating database at {kwargs['path']}')
    con = sqlite3.connect(kwargs['path']+db_name)
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

# def add_test_kwargs(**kwargs):


def update_db(pops, fits, generation, **kwargs):
    """Insert values from the pop and fits to the database"""
    print(f'Updating Database {kwargs['path']+db_name}')

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

    con = sqlite3.connect(kwargs['path'] + db_name, timeout=kwargs['update_timeout'])
    with con:
        cur = con.cursor()
        cur.executemany("INSERT INTO data VALUES(?, ?, ?, ?, ?, ?)", data)
        # con.commit()
    con.close()



if __name__ == '__main__':
    # name = 'unstable_self_rep_0'
    # fits = load_fits(**kwargs)
    #
    # pop = load_pop(0, 0, **kwargs)

    name = 'test'
    kwargs = load_kwargs(name, '../../saves/smlgp/')

    kwargs['path'] = '../../saves/smlgp/' + name + '/'



    con = sqlite3.connect(kwargs['path'] + db_name)
    cur = con.cursor()
    res = cur.execute('select gen, count(gen) from data group by gen')
    for i in res:
        print(i)
    con.close()


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


