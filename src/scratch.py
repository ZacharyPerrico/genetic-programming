"""File for testing ideas"""
import sqlite3

# import tkinter as tk
# root = tk.Tk()
# # Widgets are added here
# root.mainloop()

# from sympy import *
# import sympy

# a = sympy.Symbol('a')
# b = sympy.Symbol('b')
# c = sympy.Symbol('c')
#
# REGS = [0, a, 0, 0]
#
# REGS[2] += REGS[1]
# REGS[3] += REGS[2]
# REGS[3] //= 3
# REGS[2] *= REGS[1]
#
# REGS[2] += REGS[1]
# REGS[3] += REGS[2]
# REGS[3] //= 3
# REGS[2] *= REGS[1]
#
# REGS[2] += REGS[1]
# REGS[3] += REGS[2]
# REGS[3] //= 3
#
# print(REGS[3].simplify())


# REGS[1] *= REGS[1]
# REGS[3] += REGS[1]
# STOP
# REGS[3] += REGS[REGS[3]]

# REGS = [0, 0]
# CODE = [0] *
# # REGS[0] += REGS[REGS[0]]
# REGS[1] *= REGS[1]
# # if REGS[0] == REGS[REGS[1]]:
# #    REGS[1] -= 205
# # REGS[0] = CODE[10]
# REGS[1] += CODE[1]
# CODE[1] = REGS[1]
# # CODE[29] = REGS[1]

db_name = 'data.db'
sql_file = 'data.sql'

# con = sqlite3.connect(db_name)
# cur = con.cursor()
# cur.execute("CREATE TABLE movie(name, year, score)")
# data = [
#     ("Monty Python Live at the Hollywood Bowl", 1982, 7.9),
#     ("Monty Python's The Meaning of Life", 1983, 7.5),
#     ("Monty Python's Life of Brian", 1979, 8.0),
# ]


con = sqlite3.connect(db_name)
with open(sql_file, 'r') as f:
    init_sql = f.read()
cur = con.cursor()
cur.executescript(init_sql)
con.commit()
con.close()

data = [
    ['test 0', 1, 0, 1, 'fhweo'],
    ['test 1', 1, 0, 4, 'fheqweqwo'],
]


con = sqlite3.connect(db_name)
try:
    cur = con.cursor()
    cur.executemany("INSERT INTO data VALUES(?, ?, ?, ?, ?)", data)
    con.commit()
except:
    con.rollback()
finally:
    con.close()


with sqlite3.connect("example.db") as conn:



con = sqlite3.connect(db_name)
try:
    while True:
        i = input()
        if i == '':
            break
        res = cur.execute(i)
        print(res.fetchall())
finally:
    con.close()
    print('EXITING')



# cur.execute("CREATE TABLE data(test, seed, gen, ind, )")
# con.commit()

# res = cur.execute('PRAGMA table_info("data");')
# res = cur.execute('.schema')
# print(res.fetchall())
