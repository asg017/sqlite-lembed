# `sqlite-lembed`

A SQLite extension for generating text embeddings with [llama.cpp](https://github.com/ggerganov/llama.cpp). A sister project to [`sqlite-vec`](https://github.com/asg017/sqlite-vec) and [`sqlite-rembed`](https://github.com/asg017/sqlite-rembed). A work-in-progress!

## Usage

`sqlite-lembed` uses embeddings models that are in the [GGUF format](https://huggingface.co/docs/hub/en/gguf) to generate embeddings. These are a bit hard to find or convert, so here's a sample model you can use:

```bash
curl -L -o all-MiniLM-L6-v2.e4ce9877.q8_0.gguf https://huggingface.co/asg017/sqlite-lembed-model-examples/resolve/main/all-MiniLM-L6-v2/all-MiniLM-L6-v2.e4ce9877.q8_0.gguf
```

This is the [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model that I converted to the `.gguf` format, and quantized at `Q8_0` (made smaller at the epense of some quality).

To load it into `sqlite-lembed`, register it with the `temp.lembed_models` table.

```sql
.load ./lembed0

INSERT INTO temp.lembed_models(name, model)
  select 'all-MiniLM-L6-v2', lembed_model_from_file('all-MiniLM-L6-v2.e4ce9877.q8_0.gguf');

select lembed(
  'all-MiniLM-L6-v2',
  'The United States Postal Service is an independent agency...'
);
```

The `temp.lembed_clients` virtual table lets you "register" clients with pure `INSERT INTO` statements. The `name` field is a unique identifier for a given model, and `model` is typically provided as a path to the `.gguf` model, on disk, with the `lembed_model_from_file()` function.

### Using with `sqlite-vec`

`sqlite-lembed` works well with [`sqlite-vec`](https://github.com/asg017/sqlite-vec), a SQLite extension for vector search. Embeddings generated with `lembed()` use the same BLOB format for vectors that `sqlite-vec` uses.

Here's a sample "semantic search" application, made from a sample dataset of news article headlines.

```sql
create table articles(
  headline text
);

-- Random NPR headlines from 2024-06-04
insert into articles VALUES
  ('Shohei Ohtani''s ex-interpreter pleads guilty to charges related to gambling and theft'),
  ('The jury has been selected in Hunter Biden''s gun trial'),
  ('Larry Allen, a Super Bowl champion and famed Dallas Cowboy, has died at age 52'),
  ('After saying Charlotte, a lone stingray, was pregnant, aquarium now says she''s sick'),
  ('An Epoch Times executive is facing money laundering charge');


-- Build a vector table with embeddings of article headlines, using OpenAI's API
create virtual table vec_articles using vec0(
  headline_embeddings float[384]
);

insert into vec_articles(rowid, headline_embeddings)
  select rowid, lembed('all-MiniLM-L6-v2', headline)
  from articles;

```

Now we have a regular `articles` table that stores text headlines, and a `vec_articles` virtual table that stores embeddings of the article headlines, using the `all-MiniLM-L6-v2` model.

To perform a "semantic search" on the embeddings, we can query the `vec_articles` table with an embedding of our query, and join the results back to our `articles` table to retrieve the original headlines.

```sql
param set :query 'firearm courtroom'

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

/*
+--------------------------------------------------------------+------------------+
|                           headline                           |     distance     |
+--------------------------------------------------------------+------------------+
| Shohei Ohtani's ex-interpreter pleads guilty to charges rela | 1.14812409877777 |
| ted to gambling and theft                                    |                  |
+--------------------------------------------------------------+------------------+
| The jury has been selected in Hunter Biden's gun trial       | 1.18380105495453 |
+--------------------------------------------------------------+------------------+
| An Epoch Times executive is facing money laundering charge   | 1.27715671062469 |
+--------------------------------------------------------------+------------------+
*/
```

Notice how "firearm courtroom" doesn't appear in any of these headlines, but it can still figure out that "Hunter Biden's gun trial" is related, and the other two justice-related articles appear on top.

## Embedding Models in `.gguf` format

Most embeddings models out there are provided as PyTorch/ONNX models, but `sqlite-lembed` uses models in the [GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). However, since ggml/GGUF is relatively new, they can be hard to find. You can always [convert models yourself](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py), or here's a few pre-converted embedding models already in GGUF format:

| Model Name              | Link                                                       |
| ----------------------- | ---------------------------------------------------------- |
| `nomic-embed-text-v1.5` | https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF |
| `mxbai-embed-large-v1`  | https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1  |

## Drawbacks

1. **No batch support yet.** `llama.cpp` has support for batch processing multiple inputs, but I haven't figured that out yet. Add a :+1: to [Issue #2](https://github.com/asg017/sqlite-lembed/issues/2) if you want to see this fixed.
2. **Pre-compiled version of `sqlite-lembed` don't use the GPU.** This was done to make compiling/distrubution easier, but that means it will likely take a long time to generate embeddings. If you need it to go faster, try compiling `sqlite-lembed` yourself (docs coming soon).
