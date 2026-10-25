-- Times are Unix seconds, set by the service, so the same queries run on Postgres and on
-- the SQLite database built into the Mac app. Keep this file and
-- migrations/sqlite/0001_init.sql describing the same schema.

-- Accounts come from GitHub sign-in. GitHub tokens are never stored.
create table accounts (
    id           bigserial primary key,
    github_id    bigint not null unique,
    github_login text not null,
    created_at   bigint not null
);

-- Only the SHA-256 of a session token is stored.
create table sessions (
    token_hash bytea primary key,
    account_id bigint not null references accounts on delete cascade,
    created_at bigint not null,
    expires_at bigint not null
);
create index sessions_account on sessions (account_id);

-- GitHub device-flow logins in progress.
create table device_flows (
    id          text primary key,
    device_code text not null,
    expires_at  bigint not null
);

-- A device is identified by its iroh public key. `addr` is its latest EndpointAddr as JSON.
create table devices (
    endpoint_id text primary key,
    account_id  bigint not null references accounts on delete cascade,
    name        text not null,
    kind        text not null check (kind in ('mac', 'windows', 'other')),
    addr        text,
    last_seen   bigint,
    created_at  bigint not null,
    unique (account_id, name)
);

-- A client device allowed to send jobs to a Mac, once approved on the Mac.
create table pairings (
    id         bigserial primary key,
    client_id  text not null references devices (endpoint_id) on delete cascade,
    mac_id     text not null references devices (endpoint_id) on delete cascade,
    code       text not null,
    status     text not null check (status in ('pending', 'approved')),
    created_at bigint not null,
    expires_at bigint not null,
    unique (client_id, mac_id)
);
create index pairings_mac on pairings (mac_id);
