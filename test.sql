--.load ./dist/lembed0
--.load ../sqlite-vec/dist/vec0

.mode box
.header on
.bail on

.timer on
.echo on

select sqlite_version(), lembed_version(), vec_version();



INSERT INTO temp.lembed_models(name, model)
  select 'default', 'dist/.models/mxbai-embed-xsmall-v1-q8_0.gguf';

create table articles as
  select
    column1 as headline,
    random() % 100 as random
  from (VALUES
    ('Shohei Ohtani''s ex-interpreter pleads guilty to charges related to gambling and theft'),
    ('The jury has been selected in Hunter Biden''s gun trial'),
    ('Larry Allen, a Super Bowl champion and famed Dallas Cowboy, has died at age 52'),
    ('After saying Charlotte, a lone stingray, was pregnant, aquarium now says she''s sick'),
    ('An Epoch Times executive is facing money laundering charge'),
    ('Hassan Nasrallah’s killing transforms an already deadly regional conflict'),
    ('Who was Hassan Nasrallah, the Hezbollah leader killed by Israel?'),
    ('What is Hezbollah, the militia fighting Israel in Lebanon?'),
    ('Netanyahu defies calls for a cease-fire at the U.N., as Israel strikes Lebanon'),
    ('Death toll from Hurricane Helene mounts as aftermath assessment begins'),
    ('5 things to know from this week’s big report on cannabis'),
    ('VP debates may alter a close race’s dynamic even when they don''t predict the winner'),
    ('SpaceX launches ISS-bound crew that hopes to bring home 2 stuck astronauts'),
    ('Why the price of eggs is on the rise again'),
    ('A guide to your weekend viewing and reading'),
    ('At the border in Arizona, Harris lays out a plan to get tough on fentanyl'),
    ('A new kind of drug for schizophrenia promises fewer side effects'),
    ('Meet the astronauts preparing to travel farther from Earth than any human before'),
    ('‘SNL’ has always taken on politics. Here’s what works — and why'),
    ('Golden-age rappers make a digital-age leap — and survive'),
    ('Why Russia''s broadcaster RT turned to covertly funding American pro-Trump influencers'),
    ('Read the indictment: NYC Mayor Eric Adams charged with bribery, fraud, foreign donations'),
    ('Justice Department sues Alabama, claiming it purged voters too close to the election'),
    ('Exactly 66 years ago, another Hurricane Helene rocked the Carolinas'),
    ('A meteorologist in Atlanta rescued a woman from Helene floodwaters on camera')
  );

select
  *,
  contents,
  vec_to_json(vec_slice(embedding, 0, 8))
from lembed_batch(
  (
    select json_group_array(
      json_object(
        'id', rowid,
        'contents', headline,
        'random', random
      )
    ) from articles
  )
);


.exit
select * from articles;

.timer on
select headline, length(lembed( headline)) from articles;

select
  rowid,
  contents,
  --length(embedding),
  vec_to_json(vec_slice(embedding, 0, 8))
from lembed_batch(
  (
    select json_group_array(
      json_object(
        'id', rowid,
      'contents', headline
      )
    ) from articles
  )
);

select
  rowid,
  headline,
  vec_to_json(vec_slice(lembed(headline), 0, 8))
from articles;

.exit

select
  rowid,
  contents,
  --length(embedding),
  vec_to_json(vec_slice(embedding, 0, 8)),
  vec_to_json(vec_slice(lembed(contents), 0, 8))

from lembed_batch(
  (
    '[
      {"contents": "Shohei Ohtani''s ex-interpreter pleads guilty to charges related to gambling and theft"}
    ]'
  )
);
select
  rowid,
  contents,
  --length(embedding),
  vec_to_json(vec_slice(embedding, 0, 8)),
  vec_to_json(vec_slice(lembed(contents), 0, 8))

from lembed_batch(
  (
    '[
      {"contents": "Shohei Ohtani''s ex-interpreter pleads guilty to charges related to gambling and theft"},
      {"contents": "The jury has been selected in Hunter Biden''s gun trial"}
    ]'
  )
);


.exit

select vec_length(lembed('all-MiniLM-L6-v2', 'hello')) as embedding;

create table shows as
  select
    value as name,
    lembed('all-MiniLM-L6-v2', value) as embedding
  from json_each('[
    "family guy",
    "american dad",
    "simpsons",
    "channel 7 news",
    "modern family"
  ]');
--select lembed('all-MiniLM-L6-v2', 'family guy');
select
  s1.name,
  s2.name,
  vec_distance_l2(s1.embedding, s2.embedding)
from shows as s1
join shows as s2 on s1.rowid < s2.rowid
order by 3;
.exit

INSERT INTO temp.lembed_models(name, model, model_options, context_options)
with models as (
  select
    column1 as name,
    column2 as model_path
  from (VALUES
    ('nomic-1.5-f32',  'models/nomic-embed-text-v1.5.f32.gguf'),
    ('nomic-1.5-f16',  'models/nomic-embed-text-v1.5.f16.gguf'),
    ('nomic-1.5-Q8_0', 'models/nomic-embed-text-v1.5.Q8_0.gguf'),
    ('nomic-1.5-Q4_0', 'models/nomic-embed-text-v1.5.Q4_0.gguf'),
    ('nomic-1.5-Q2_K', 'models/nomic-embed-text-v1.5.Q2_K.gguf')
  )
)
select
  name,
  lembed_model_from_file(model_path),
  lembed_model_options(
    'n_gpu_layers', 99
  ),
  lembed_context_options(
    'n_ctx', 8192,
    'rope_scaling_type', 'yarn',
    'rope_freq_scale', .75
  )
from models;

select
  rowid,
  name,
  model,
  lembed_model_size(model)
from temp.lembed_models;


create table documents as
  select
    value as contents
  from json_each('[
    "alex garcia",
    "Mars is the fourth planet from the Sun. It was formed approximately 4.5 billion years ago"
  ]');

create table document_embeddings_F32 as
 select rowid, lembed('nomic-1.5', 'search_document: ' || contents) as embedding from documents;
create table document_embeddings_F16 as
 select rowid, lembed('nomic-1.5-f16', 'search_document: ' || contents) as embedding from documents;
create table document_embeddings_Q8_0 as
 select rowid, lembed('nomic-1.5-Q8_0', 'search_document: ' || contents) as embedding from documents;
create table document_embeddings_Q4_0 as
 select rowid, lembed('nomic-1.5-Q4_0', 'search_document: ' || contents) as embedding from documents;
create table document_embeddings_Q2_K as
 select rowid, lembed('nomic-1.5-Q2_K', 'search_document: ' || contents) as embedding from documents;

select
  contents,
  vec_distance_l2(F32.embedding, F16.embedding) as f32_f16,
  vec_distance_l2(F32.embedding, Q8_0.embedding) as f32_q80,
  vec_distance_l2(F32.embedding, Q4_0.embedding) as f32_q40,
  vec_distance_l2(F32.embedding, Q2_K.embedding) as f32_q2k
from documents
left join document_embeddings_F32 as F32 on F32.rowid == documents.rowid
left join document_embeddings_F16 as F16 on F16.rowid == documents.rowid
left join document_embeddings_Q8_0 as Q8_0 on Q8_0.rowid == documents.rowid
left join document_embeddings_Q4_0 as Q4_0 on Q4_0.rowid == documents.rowid
left join document_embeddings_Q2_K as Q2_K on Q2_K.rowid == documents.rowid
;

.exit

--select llama_load_default_model('nomic-embed-text-v1.f32.gguf');
--select llama_load_default_model('nomic-embed-text-v1.Q2_K.gguf');
--select llama_load_default_model('all-MiniLM-L6-v2-44eb4044.gguf');
--select lembed_load_default_model('models/nomic-embed-text-v1.5.f32.gguf');
select lembed_load_default_model('/Users/alex/projects/research-sqlite-llama-embeddings/vendor/llama.cpp/models/ggml-model-Q2_K.gguf');



select vec_to_json(
  vec_normalize(
    vec_slice(lembed("search_document: alex garcia"), 0, 256)
  )
);

select lembed_debug();

.exit

--.exit
/*

INSERT INTO llama_models(name, model)
VALUES ('nomic-embed-text-v1', llama_model_from_file('nomic-embed-text-v1.Q2_K.gguf'));;

select llama_embed('nomic-embed-text-v1', 'my name is...');
*/

.param set :prompt 'alex garcia simon willison 1 2 3 4     5.'

select
  value,
  lembed_token_type(value),
  lembed_token_to_piece(value)
from json_each(
  lembed_tokenize_json(:prompt)
);

--select llama_split_debug(:prompt, 20, 0);
--select llama_split_debug(:prompt, 5, 0);

select
  rowid,
  contents,
  token_count,
  lembed_tokenize_json(contents)
from lembed_chunks(
  --'alex garcia is from los angeles california.'
  'alex garcia simon willison 1 2 3 4     5 '
);

.exit

select lembed_tokenize_json('hi i am ALEX alex garcia');
select
  value,
  --llama_token_score(value),
  lembed_token_type(value),
  lembed_token_to_piece(value)
from json_each(
  lembed_tokenize_json(
    'hi i am alex ALEX garcia.

    1 2 3 4 5 6 7 8 9 10
    '
  )
);
