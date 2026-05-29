--DROP TABLE IF EXISTS data;

CREATE TABLE IF NOT EXISTS data (
  test TEXT,
  seed INT NOT NULL,
  gen INT,
  id INT,
  fitness REAL,
  data TEXT,
  PRIMARY KEY (test, seed, gen, id)
);

CREATE TABLE IF NOT EXISTS kwargs (
    test TEXT,
    PRIMARY KEY (test)
)