--DROP TABLE IF EXISTS data;

CREATE TABLE IF NOT EXISTS data (
  test TEXT,
  seed ANY,
  gen INT,
  id INT,
  fit REAL,
  data TEXT,
  PRIMARY KEY (test, seed, gen, id)
) STRICT;

CREATE TABLE IF NOT EXISTS kwargs (
    test TEXT,
    PRIMARY KEY (test)
)