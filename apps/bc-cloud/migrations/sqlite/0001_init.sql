-- SQLite version of migrations/postgres/0001_init.sql, used when the Mac app hosts the
-- cloud. Keep both files describing the same schema.

create table accounts (
    id           integer primary key autoincrement,
    github_id    integer not null unique,
    github_login text not null,
    created_at   integer not null
);

create table sessions (
    token_hash blob primary key,
    account_id integer not null references accounts (id) on delete cascade,
    created_at integer not null,
    expires_at integer not null
);
create index sessions_account on sessions (account_id);

create table device_flows (
    id          text primary key,
    device_code text not null,
    expires_at  integer not null
);

create table devices (
    endpoint_id text primary key,
    account_id  integer not null references accounts (id) on delete cascade,
    name        text not null,
    kind        text not null check (kind in ('mac', 'windows', 'other')),
    addr        text,
    last_seen   integer,
    created_at  integer not null,
    unique (account_id, name)
);

create table pairings (
    id         integer primary key autoincrement,
    client_id  text not null references devices (endpoint_id) on delete cascade,
    mac_id     text not null references devices (endpoint_id) on delete cascade,
    code       text not null,
    status     text not null check (status in ('pending', 'approved')),
    created_at integer not null,
    expires_at integer not null,
    unique (client_id, mac_id)
);
create index pairings_mac on pairings (mac_id);
