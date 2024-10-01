.bail on

.load ./dist/lembed0
.load ../sqlite-vec/dist/vec0

select lembed_version(), lembed_debug();

insert into temp.lembed_models(name, model)
  --select 'default', lembed_model_from_file('/Users/alex/projects/llama.cpp/all-MiniLM-L6-v2.F16.gguf');
  --select 'default', lembed_model_from_file('./all-MiniLM-L6-v2.e4ce9877.f32.gguf');
  --select 'default', lembed_model_from_file('./all-MiniLM-L6-v2.F32.gguf');
  select 'default', lembed_model_from_file('all-MiniLM-L6-v2.Q6_K.gguf');

--select length(lembed('asdf'));
.mode box
.header on
.timer on

select
  rowid,
  --contents,
  typeof(embedding),
  quote(substr(embedding, 0, 8))
  --vec_to_json(vec_slice(embedding, 0, 4))
from lembed_batch(
  (
   select json_group_array(
      json_object('contents', headline)
   )
   from (select * from articles limit 1000)
  )
);

select sum(length(lembed(headline))) from (select * from articles limit 1000);

select
  rowid,
  --contents,
  typeof(embedding),
  quote(substr(embedding, 0, 8))
  --vec_to_json(vec_slice(embedding, 0, 4))
from lembed_batch(
  (
   select json_group_array(
      json_object('contents', headline)
   )
   from (select * from articles limit 1000)
  )
);
