"""File for testing ideas"""
import sqlite3

db_name = 'data.db'




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
