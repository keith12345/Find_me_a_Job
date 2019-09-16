import json

import psycopg2


with open('../conn.json') as fp:
    conn_kwargs = json.load(fp)

conn = psycopg2.connect(**conn_kwargs)
cur = conn.cursor()

create_tables = '''
CREATE TABLE job_listings_pages (
    date_pulled DATE NOT NULL,
    query varchar(255) NOT NULL,
    city varchar(255) NOT NULL,
    link varchar(2500) NOT NULL
);

CREATE TABLE jobs_pages (
    date_pulled DATE NOT NULL,
    query varchar(255) NOT NULL,
    city varchar(255) NOT NULL,
    link varchar(2500) NOT NULL
);
'''

cur.execute(create_tables)

conn.commit()

conn.close()