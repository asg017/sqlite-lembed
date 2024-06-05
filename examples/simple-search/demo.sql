.bail on
.mode table
.header on
.timer on
.echo on

.load ../../dist/lembed0
.load ../../../sqlite-vec/dist/vec0

INSERT INTO temp.lembed_models(name, model)
  select 'all-MiniLM-L6-v2', lembed_model_from_file('all-MiniLM-L6-v2.e4ce9877.q8_0.gguf');
  --select 'all-MiniLM-L6-v2', lembed_model_from_file('../../models/nomic-embed-text-v1.5.f16.gguf');


create table articles(headline text);


-- Random NPR headlines from 2024-06-04
insert into articles VALUES
  ('Shohei Ohtani''s ex-interpreter pleads guilty to charges related to gambling and theft'),
  ('The jury has been selected in Hunter Biden''s gun trial'),
  ('Larry Allen, a Super Bowl champion and famed Dallas Cowboy, has died at age 52'),
  ('After saying Charlotte, a lone stingray, was pregnant, aquarium now says she''s sick'),
  ('An Epoch Times executive is facing money laundering charge');


-- Seed a vector table with embeddings of article headlines
create virtual table vec_articles using vec0(headline_embeddings float[384]);


insert into vec_articles(rowid, headline_embeddings)
  select rowid, lembed('all-MiniLM-L6-v2', headline)
  from articles;


.param set :query 'firearm courtroom'

with matches as (
  select
    rowid,
    distance
  from vec_articles
  where headline_embeddings match lembed('all-MiniLM-L6-v2', :query)
  order by distance
  limit 3
)
select
  headline,
  distance
from matches
left join articles on articles.rowid = matches.rowid;
