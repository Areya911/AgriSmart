-- migrate_add_users.sql
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,           -- change to hashed password in production
  email TEXT,
  phone TEXT,
  state TEXT,
  district TEXT,
  language TEXT DEFAULT 'en',
  land_size_acres REAL,
  farming_type TEXT,                -- 'Beginner'|'Middle'|'Advanced'
  created_at TEXT NOT NULL,
  updated_at TEXT
);

-- optional: add an index for location lookups
CREATE INDEX IF NOT EXISTS idx_users_location ON users(state, district);
