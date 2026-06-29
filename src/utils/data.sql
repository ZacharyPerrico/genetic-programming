--DROP TABLE IF EXISTS data;

CREATE TABLE IF NOT EXISTS data (
  test TEXT,
  seed ANY,
  gen INT,
  ind INT,
  fit REAL,
  genotype TEXT,
  PRIMARY KEY (test, seed, gen, ind)
) STRICT;