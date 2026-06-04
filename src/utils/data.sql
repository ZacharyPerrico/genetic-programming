--DROP TABLE IF EXISTS data;

CREATE TABLE IF NOT EXISTS data (
  test TEXT,
  seed INT,
  gen INT,
  id INT,
  fit REAL,
  data TEXT,
  PRIMARY KEY (test, seed, gen, id)
);

CREATE TABLE IF NOT EXISTS kwargs (
    test TEXT,
    PRIMARY KEY (test)
)