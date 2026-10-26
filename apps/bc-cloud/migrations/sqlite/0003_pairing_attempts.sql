-- Wrong pairing codes typed on the Mac. A request is deleted after the third, so the
-- code cannot be guessed. Keep the postgres and sqlite versions describing the same schema.
alter table pairings add column attempts integer not null default 0;
