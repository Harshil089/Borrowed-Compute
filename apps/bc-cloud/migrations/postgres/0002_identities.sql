-- An account can now be reached through several sign-in providers (GitHub, Google).
-- Each provider identity points at one account; identities whose providers report the
-- same verified email address share an account. Keep this file and
-- migrations/sqlite/0002_identities.sql describing the same schema.

create table identities (
    provider   text not null check (provider in ('github', 'google')),
    -- The provider's stable user id (GitHub's numeric id, Google's `sub`).
    subject    text not null,
    account_id bigint not null references accounts on delete cascade,
    -- Only ever a verified address; null until the provider reports one.
    email      text,
    created_at bigint not null,
    primary key (provider, subject)
);
create index identities_account on identities (account_id);
create index identities_email on identities (email);

insert into identities (provider, subject, account_id, email, created_at)
    select 'github', github_id::text, id, null, created_at from accounts;

alter table accounts rename column github_login to login;
alter table accounts drop column github_id;

alter table device_flows add column provider text not null default 'github';
