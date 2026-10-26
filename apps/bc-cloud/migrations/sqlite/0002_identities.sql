-- SQLite version of migrations/postgres/0002_identities.sql. Keep both describing the
-- same schema.
--
-- SQLite cannot drop a NOT NULL column, so `accounts` is rebuilt. Dropping the old table
-- with foreign keys enforced would cascade-delete every session and device, so
-- `bc_cloud::connect` runs migrations on a connection with foreign keys off, then checks
-- them with `pragma foreign_key_check`. sqlx wraps this file in a transaction.

create table accounts_new (
    id         integer primary key autoincrement,
    login      text not null,
    created_at integer not null
);
insert into accounts_new (id, login, created_at) select id, github_login, created_at from accounts;

create table identities (
    provider   text not null check (provider in ('github', 'google')),
    subject    text not null,
    account_id integer not null references accounts (id) on delete cascade,
    email      text,
    created_at integer not null,
    primary key (provider, subject)
);
insert into identities (provider, subject, account_id, email, created_at)
    select 'github', cast(github_id as text), id, null, created_at from accounts;
create index identities_account on identities (account_id);
create index identities_email on identities (email);

drop table accounts;
alter table accounts_new rename to accounts;

alter table device_flows add column provider text not null default 'github';
