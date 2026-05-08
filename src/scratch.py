"""File for testing ideas"""
import sqlite3

# from sympy import *
import sympy

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

# db_name = 'data.db'




# con = sqlite3.connect(db_name)
# cur = con.cursor()
# cur.execute("CREATE TABLE movie(name, year, score)")
# data = [
#     ("Monty Python Live at the Hollywood Bowl", 1982, 7.9),
#     ("Monty Python's The Meaning of Life", 1983, 7.5),
#     ("Monty Python's Life of Brian", 1979, 8.0),
# ]
# cur.executemany("INSERT INTO movie VALUES(?, ?, ?)", data)
# con.commit()
# con.close()

# while True:
#     res = cur.execute(input())
#     print(res.fetchall)
