.load ./dist/lembed0
.load ../sqlite-vec/dist/vec0

.mode box
.header on
.bail on

.timer on
.echo on

select lembed_version(), lembed_debug();



INSERT INTO temp.lembed_models(name, model)
  select 'all-MiniLM-L6-v2', lembed_model_from_file('models/all-MiniLM-L6-v2-44eb4044.gguf');

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
  lembed_model_from_file(model_path),`
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
