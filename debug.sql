.load ./dist/lembed0
.load ../sqlite-vec/dist/vec0

select lembed_version(), lembed_debug();

insert into temp.lembed_models(name, model)
  select 'default', lembed_model_from_file('');

select vec_to_json(vec_slice(lembed('Shohei Ohtani''s ex-interpreter pleads guilty to charges related to gambling and theft'), 0, 8));
select vec_to_json(vec_slice(lembed('The jury has been selected in Hunter Biden''s gun trial'), 0, 8));


.mode box
.header on

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
