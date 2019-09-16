import json

params = {
    'host': 'scraperstorage.ccgdvksfhbwt.us-east-2.rds.amazonaws.com',
    'user': 'Keith12345',
    'password': 'Password',
    'port': 5432,
    'dbname': 'jobs_links'
}

with open('conn.json', 'w') as fp:
    json.dump(params, fp)