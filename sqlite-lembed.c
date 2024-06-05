#include "sqlite-lembed.h"
#include "llama.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(X) (void)(X)
#endif

void dummy_log(enum ggml_log_level level, const char *text, void *user_data) {}

static void normalize(float *vec, float *out, int n) {
  float norm = 0;
  for (int i = 0; i < n; i++) {
    norm += vec[i] * vec[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < n; i++) {
    out[i] = vec[i] / norm;
  }
}

#define LEMBED_TOKEN_SUBTYPE 116 // ascii 't'

int tokenize(struct llama_model *model, const char *input, size_t input_length,
             int *token_count, llama_token **tokens) {
  int input_token_count_estimate =
      llama_tokenize(model, input, input_length, NULL, 0, true, true);
  if (input_token_count_estimate >= 0) {
    return SQLITE_ERROR;
  }
  *tokens =
      sqlite3_malloc(sizeof(llama_token) * abs(input_token_count_estimate));
  if (!(*tokens)) {
    return SQLITE_NOMEM;
  }
  int input_token_count =
      llama_tokenize(model, input, input_length, *tokens,
                     abs(input_token_count_estimate), true, true);
  if (input_token_count != abs(input_token_count_estimate)) {
    sqlite3_free(*tokens);
    return SQLITE_ERROR;
  }

  *token_count = input_token_count;
  return SQLITE_OK;
}

int embed_single(struct llama_model *model, struct llama_context *context,
                 const char *input, size_t input_length,
                 /** Output float embedding */
                 float **out_embedding,
                 /** Output embedding length (n dimensions) */
                 int *out_dimensions) {
  int n_batch = 512;
  int n_ctx_train = llama_n_ctx_train(model);
  int n_ctx = llama_n_ctx(context);

  llama_token *tokens;
  int token_count;
  int rc = tokenize(model, input, input_length, &token_count, &tokens);
  if(rc != SQLITE_OK) {
    // TODO error message
    return rc;
  }

  struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

  int seq_id = 0;
  // llama_batch_add(batch, tokens, 0, )
  for (int i = 0; i < token_count; i++) {
    batch.token[batch.n_tokens] = tokens[i];
    batch.pos[batch.n_tokens] = i;

    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;

    batch.logits[batch.n_tokens] = i == (token_count - 1);
    batch.n_tokens++;
  }

  int dimensions = llama_n_embd(model);
  float *output_embedding = sqlite3_malloc(sizeof(float) * dimensions);
  if(!output_embedding) {
    llama_batch_free(batch);
    return SQLITE_NOMEM;
  }

  llama_kv_cache_clear(context); // KV not needed for embeddings?
  rc = llama_decode(context, batch);
  if(rc != 0) {
    sqlite3_free(output_embedding);
    llama_batch_free(batch);
    return SQLITE_ERROR;
  }

  float * source_embedding;
  if(llama_pooling_type(context) == LLAMA_POOLING_TYPE_NONE) {
    source_embedding = llama_get_embeddings(context);
  }
  else {
    source_embedding = llama_get_embeddings_seq(context, batch.seq_id[0][0]);
  }
  if(!source_embedding) {
    sqlite3_free(output_embedding);
    llama_batch_free(batch);
    return SQLITE_ERROR;
  }

  normalize(source_embedding, output_embedding, dimensions);
  llama_batch_free(batch);

  *out_dimensions = dimensions;
  *out_embedding = output_embedding;
  return SQLITE_OK;
}

typedef struct ApiModel ApiModel;
struct ApiModel {
  char *name;
  struct llama_model *model;
  struct llama_context *context;
};

#define MAX_MODELS 16
struct Api {
  int default_index;
  ApiModel models[MAX_MODELS];
};

void api_free(void *p) {
  struct Api *a = (struct Api *)p;
  llama_backend_free();
  sqlite3_free(a);
}

typedef struct lembed_model_options lembed_model_options;
struct lembed_model_options {
  int32_t n_gpu_layers;

  int8_t defined[1];
};
static char *POINTER_NAME_MODEL = "lembed_model";
static char *POINTER_NAME_MODEL_OPTIONS = "lembed_model_options";

static void lembed_model_size(sqlite3_context *context, int argc,
                              sqlite3_value **argv) {
  struct llama_model *model =
      sqlite3_value_pointer(argv[0], POINTER_NAME_MODEL);
  if (!model)
    return;
  sqlite3_result_int64(context, llama_model_size(model));
}

static void lembed_model_options_(sqlite3_context *context, int argc,
                                  sqlite3_value **argv) {
  assert(argc >= 0);
  assert(argc % 2 == 0);
  lembed_model_options *o = sqlite3_malloc(sizeof(lembed_model_options));
  assert(o);
  memset(o, 0, sizeof(*o));

  for (int i = 0; i < argc; i += 2) {
    sqlite3_value *key = argv[i];
    sqlite3_value *value = argv[i + 1];
    assert(sqlite3_value_type(key) == SQLITE_TEXT);
    const char *k = (const char *)sqlite3_value_text(key);
    if (sqlite3_stricmp(k, "n_gpu_layers") == 0) {
      o->n_gpu_layers = sqlite3_value_int(value);
      o->defined[0] = 1;
    } else {
      abort();
    }
  }
  sqlite3_result_pointer(context, o, POINTER_NAME_MODEL_OPTIONS, sqlite3_free);
}

typedef struct lembed_context_options lembed_context_options;
struct lembed_context_options {
  uint32_t seed;
  uint32_t n_ctx;
  enum llama_rope_scaling_type rope_scaling_type;
  float rope_freq_scale;

  int8_t defined[4];
};
static char *POINTER_NAME_CONTEXT_OPTIONS = "lembed_context_options";

static void lembed_context_options_(sqlite3_context *context, int argc,
                                    sqlite3_value **argv) {
  assert(argc >= 0);
  assert(argc % 2 == 0);
  lembed_context_options *o = sqlite3_malloc(sizeof(lembed_context_options));
  assert(o);
  memset(o, 0, sizeof(*o));

  for (int i = 0; i < argc; i += 2) {
    sqlite3_value *key = argv[i];
    sqlite3_value *value = argv[i + 1];
    assert(sqlite3_value_type(key) == SQLITE_TEXT);
    const char *k = (const char *)sqlite3_value_text(key);
    if (sqlite3_stricmp("seed", k) == 0) {
      sqlite3_int64 v = sqlite3_value_int64(value);
      assert(v > 0);
      o->seed = v;
      o->defined[0] = 1;
    } else if (sqlite3_stricmp("n_ctx", k) == 0) {
      sqlite3_int64 v = sqlite3_value_int64(value);
      assert(v > 0);
      o->n_ctx = v;
      o->defined[1] = 1;
    } else if (sqlite3_stricmp("rope_scaling_type", k) == 0) {
      const char *v = (const char *)sqlite3_value_text(value);
      if (sqlite3_stricmp(v, "none")) {
        o->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
      } else if (sqlite3_stricmp(v, "linear")) {
        o->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
      } else if (sqlite3_stricmp(v, "yarn")) {
        o->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
      } else {
        abort();
      }

      o->defined[2] = 1;
    } else if (sqlite3_stricmp(k, "rope_freq_scale") == 0) {
      o->rope_freq_scale = sqlite3_value_double(value);
      o->defined[3] = 1;
    } else {
      abort();
    }
  }
  sqlite3_result_pointer(context, o, POINTER_NAME_CONTEXT_OPTIONS,
                         sqlite3_free);
}
static char *POINTER_NAME_MODEL_PATH = "lembed_model_path";

static void lembed_model_from_file(sqlite3_context *context, int argc,
                                   sqlite3_value **argv) {
  sqlite3_result_pointer(context,
                         sqlite3_mprintf("%.*s", sqlite3_value_bytes(argv[0]),
                                         sqlite3_value_text(argv[0])),
                         POINTER_NAME_MODEL_PATH, sqlite3_free);
}


static void _static_text_func(sqlite3_context *context, int argc,
                              sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  sqlite3_result_text(context, sqlite3_user_data(context), -1, SQLITE_STATIC);
}

int api_model_from_name(struct Api *api, const char *name, int name_length,
                        struct llama_model **model,
                        struct llama_context **context) {
  for (int i = 0; i < MAX_MODELS; i++) {
    if (!api->models[i].name)
      continue;
    if (strncmp(api->models[i].name, name, name_length) == 0) {
      *model = api->models[i].model;
      if (context)
        *context = api->models[i].context;
      return SQLITE_OK;
    }
  }
  return SQLITE_ERROR;
}
static void lembed(sqlite3_context *context, int argc, sqlite3_value **argv) {
  struct llama_model *model;
  struct llama_context *ctx;
  int rc = api_model_from_name((struct Api *)sqlite3_user_data(context),
                               (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, &ctx);
  if(rc != SQLITE_OK) {
    sqlite3_result_error(context, "Unknown model name. Was it registered with lembed_models?", -1);
    return;
  }
  const char *input = (const char *)sqlite3_value_text(argv[1]);
  sqlite3_int64 input_len = sqlite3_value_bytes(argv[1]);
  int dimensions;
  float *embedding;
  rc = embed_single(model, ctx, input, input_len, &embedding, &dimensions);
  if(rc != SQLITE_OK) {
    sqlite3_result_error(context, "Error generating embedding", -1);
    return;
  }
  sqlite3_result_blob(context, embedding, sizeof(float) * dimensions, sqlite3_free);
  sqlite3_result_subtype(context, 223); // TODO define
}

static void lembed_tokenize_json(sqlite3_context *context, int argc,
                                 sqlite3_value **argv) {
  struct llama_model *model;
  int rc = api_model_from_name((struct Api *)sqlite3_user_data(context),
                               (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, NULL);
  const char *input = (const char *)sqlite3_value_text(argv[1]);
  sqlite3_int64 input_len = sqlite3_value_bytes(argv[1]);
  int token_count;
  llama_token *tokens;
  rc = tokenize(model, input, input_len, &token_count, &tokens);
  assert(rc == SQLITE_OK);

  sqlite3_str *s = sqlite3_str_new(NULL);
  sqlite3_str_appendchar(s, 1, '[');
  for (int i = 0; i < token_count; i++) {
    if (i != 0) {
      sqlite3_str_appendchar(s, 1, ',');
    }
    sqlite3_str_appendf(s, "%d", tokens[i]);
  }
  sqlite3_str_appendchar(s, 1, ']');
  char *result = sqlite3_str_finish(s);
  assert(result);
  sqlite3_result_text(context, result, -1, sqlite3_free);
}

static void lembed_token_score(sqlite3_context *context, int argc,
                               sqlite3_value **argv) {
  struct llama_model *model;
  int rc = api_model_from_name((struct Api *)sqlite3_user_data(context),
                               (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, NULL);

  int32_t token = sqlite3_value_int(argv[1]);

  float score = llama_token_get_score(model, token);
  sqlite3_result_double(context, score);
}
static void lembed_token_to_piece_(sqlite3_context *context, int argc,
                                   sqlite3_value **argv) {
  struct llama_model *model;
  int rc = api_model_from_name((struct Api *)sqlite3_user_data(context),
                               (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, NULL);

  int32_t token = sqlite3_value_int(argv[1]);
#define BUFLEN 256
  char buf[BUFLEN];
  int n = llama_token_to_piece(model, token, buf, BUFLEN, false);
  if (n) {
    sqlite3_result_text(context, buf, n, SQLITE_TRANSIENT);
  } else {
    sqlite3_result_null(context);
  }
}

static void _noop(sqlite3_context *context, int argc, sqlite3_value **argv) {}
static void ggml_test(sqlite3_context *context, int argc,
                      sqlite3_value **argv) {
  sqlite3_result_int64(context, ggml_cpu_has_avx());
}

#pragma region lembed_models() table function

typedef struct lembed_models_vtab lembed_models_vtab;
struct lembed_models_vtab {
  sqlite3_vtab base;
  struct Api *api;
};

typedef struct lembed_models_cursor lembed_models_cursor;
struct lembed_models_cursor {
  sqlite3_vtab_cursor base;
  sqlite3_int64 iRowid;
};

static int lembed_modelsConnect(sqlite3 *db, void *pAux, int argc,
                                const char *const *argv, sqlite3_vtab **ppVtab,
                                char **pzErr) {
  lembed_models_vtab *pNew;
  int rc;
  if (strcmp(argv[1], "temp") != 0) {
    // return SQLITE_ERROR;
  }
#define LEMBED_MODELS_NAME            0
#define LEMBED_MODELS_MODEL           1
#define LEMBED_MODELS_MODEL_OPTIONS   2
#define LEMBED_MODELS_CONTEXT_OPTIONS 3
  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(name, model, model_options "
                                "hidden, context_options hidden)");
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
    pNew->api = pAux;
  }
  return rc;
}

static int lembed_modelsDisconnect(sqlite3_vtab *pVtab) {
  lembed_models_vtab *p = (lembed_models_vtab *)pVtab;
  sqlite3_free(p);
  return SQLITE_OK;
}

#define POINTER_SUBTYPE 112

static int lembed_modelsUpdate(sqlite3_vtab *pVTab, int argc,
                               sqlite3_value **argv, sqlite_int64 *pRowid) {
  lembed_models_vtab *p = (lembed_models_vtab *)pVTab;
  // DELETE operation
  if (argc == 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    return SQLITE_ERROR;
  }
  // INSERT operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    sqlite3_value **columnValues = &argv[2];
    const char *key =
        (const char *)sqlite3_value_text(columnValues[LEMBED_MODELS_NAME]);
    int idx = -1;
    for (int i = 0; i < MAX_MODELS; i++) {
      if (!p->api->models[i].name) {
        p->api->models[i].name = sqlite3_mprintf("%s", key);
        idx = i;
        break;
      }
    }
    if (idx < 0)
      abort();

    const char *modelPath = sqlite3_value_pointer(
        columnValues[LEMBED_MODELS_MODEL], POINTER_NAME_MODEL_PATH);
    assert(modelPath);

    lembed_model_options *modelOptions = NULL;
    if (sqlite3_value_subtype(columnValues[LEMBED_MODELS_MODEL_OPTIONS]) ==
        POINTER_SUBTYPE) {
      modelOptions =
          sqlite3_value_pointer(columnValues[LEMBED_MODELS_MODEL_OPTIONS],
                                POINTER_NAME_MODEL_OPTIONS);
    }

    lembed_context_options *contextOptions = NULL;
    if (sqlite3_value_subtype(columnValues[LEMBED_MODELS_CONTEXT_OPTIONS]) ==
        POINTER_SUBTYPE) {
      contextOptions =
          sqlite3_value_pointer(columnValues[LEMBED_MODELS_CONTEXT_OPTIONS],
                                POINTER_NAME_CONTEXT_OPTIONS);
    }

    struct llama_model *model;
    struct llama_model_params mparams = llama_model_default_params();
    if (modelOptions && modelOptions->defined[0]) {
      mparams.n_gpu_layers = modelOptions->n_gpu_layers;
    }

    model = llama_load_model_from_file(modelPath, mparams);
    if (!model) {
      return SQLITE_ERROR;
    }

    struct llama_context *ctx;
    struct llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = 1;
    if (contextOptions) {
      if (contextOptions->defined[0]) {
        cparams.seed = contextOptions->seed;
      }
      if (contextOptions->defined[1]) {
        cparams.n_ctx = contextOptions->n_ctx;
      }
      if (contextOptions->defined[2]) {
        cparams.rope_scaling_type = contextOptions->rope_scaling_type;
      }
      if (contextOptions->defined[3]) {
        cparams.rope_freq_scale = contextOptions->rope_freq_scale;
      }
    }

    ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
      llama_free_model(model);
      return SQLITE_ERROR;
    }
    p->api->models[idx].model = model;
    p->api->models[idx].context = ctx;

    if (strcmp(key, "default") == 0) {
      printf("default detected\n");
    }
    return SQLITE_OK;
  }
  // UPDATE operation
  else if (argc > 1 && sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    if ((sqlite3_value_type(argv[0]) == SQLITE_INTEGER) &&
        (sqlite3_value_type(argv[1]) == SQLITE_INTEGER) &&
        (sqlite3_value_int64(argv[0]) == sqlite3_value_int64(argv[1]))) {
      return SQLITE_ERROR;
    }

    return SQLITE_ERROR;
  }
  return SQLITE_ERROR;
}

static int lembed_modelsOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  lembed_models_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int lembed_modelsClose(sqlite3_vtab_cursor *cur) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int lembed_modelsBestIndex(sqlite3_vtab *pVTab,
                                  sqlite3_index_info *pIdxInfo) {
  pIdxInfo->idxNum = 1;
  pIdxInfo->estimatedCost = (double)10;
  pIdxInfo->estimatedRows = 10;
  return SQLITE_OK;
}

static int lembed_modelsNext(sqlite3_vtab_cursor *cur);
static int lembed_modelsFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                               const char *idxStr, int argc,
                               sqlite3_value **argv) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)pVtabCursor;
  struct Api *api = ((lembed_models_vtab *)pVtabCursor->pVtab)->api;
  pCur->iRowid = -1;
  lembed_modelsNext(pVtabCursor);
  return SQLITE_OK;
}

static int lembed_modelsRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

static int lembed_modelsNext(sqlite3_vtab_cursor *cur) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  lembed_models_vtab *p = (lembed_models_vtab *)pCur->base.pVtab;
  pCur->iRowid++;
  while (pCur->iRowid < MAX_MODELS) {
    if (p->api->models[pCur->iRowid].name) {
      return SQLITE_OK;
    }
    pCur->iRowid++;
  }
  return SQLITE_OK;
}

static int lembed_modelsEof(sqlite3_vtab_cursor *cur) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  return pCur->iRowid >= MAX_MODELS;
}

static int lembed_modelsColumn(sqlite3_vtab_cursor *cur,
                               sqlite3_context *context, int i) {
  lembed_models_cursor *pCur = (lembed_models_cursor *)cur;
  lembed_models_vtab *p = (lembed_models_vtab *)cur->pVtab;
  switch (i) {
  case LEMBED_MODELS_NAME:
    sqlite3_result_text(context, p->api->models[pCur->iRowid].name, -1,
                        SQLITE_TRANSIENT);
    break;
  case LEMBED_MODELS_MODEL:
    sqlite3_result_pointer(context, p->api->models[pCur->iRowid].model,
                           POINTER_NAME_MODEL, NULL);
    break;
  }
  return SQLITE_OK;
}

static sqlite3_module lembed_modelsModule = {
    /* iVersion    */ 3,
    /* xCreate     */ 0,
    /* xConnect    */ lembed_modelsConnect,
    /* xBestIndex  */ lembed_modelsBestIndex,
    /* xDisconnect */ lembed_modelsDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ lembed_modelsOpen,
    /* xClose      */ lembed_modelsClose,
    /* xFilter     */ lembed_modelsFilter,
    /* xNext       */ lembed_modelsNext,
    /* xEof        */ lembed_modelsEof,
    /* xColumn     */ lembed_modelsColumn,
    /* xRowid      */ lembed_modelsRowid,
    /* xUpdate     */ lembed_modelsUpdate,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0};
#pragma endregion

#pragma region lembed_chunks() table function

typedef struct lembed_chunks_vtab lembed_chunks_vtab;
struct lembed_chunks_vtab {
  sqlite3_vtab base;
  struct Api *api;
};

typedef struct lembed_chunks_cursor lembed_chunks_cursor;
struct lembed_chunks_cursor {
  sqlite3_vtab_cursor base;
  sqlite3_int64 iRowid;
  int32_t chunks_count;
  char **chunks;
};

static int lembed_chunksConnect(sqlite3 *db, void *pAux, int argc,
                                const char *const *argv, sqlite3_vtab **ppVtab,
                                char **pzErr) {
  lembed_chunks_vtab *pNew;
  int rc;
#define lembed_chunks_CONTENTS 0
#define lembed_chunks_TOKEN_COUNT 1
#define lembed_chunks_SOURCE 2
#define lembed_chunks_CHUNK_SIZE 3
  rc = sqlite3_declare_vtab(db, "CREATE TABLE x(contents, token_count, source "
                                "hidden, chunk_size hidden)");
  if (rc == SQLITE_OK) {
    pNew = sqlite3_malloc(sizeof(*pNew));
    *ppVtab = (sqlite3_vtab *)pNew;
    if (pNew == 0)
      return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
    pNew->api = pAux;
  }
  return rc;
}

static int lembed_chunksDisconnect(sqlite3_vtab *pVtab) {
  lembed_chunks_vtab *p = (lembed_chunks_vtab *)pVtab;
  sqlite3_free(p);
  return SQLITE_OK;
}

static int lembed_chunksOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  lembed_chunks_cursor *pCur;
  pCur = sqlite3_malloc(sizeof(*pCur));
  if (pCur == 0)
    return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int lembed_chunksClose(sqlite3_vtab_cursor *cur) {
  lembed_chunks_cursor *pCur = (lembed_chunks_cursor *)cur;
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int lembed_chunksBestIndex(sqlite3_vtab *pVTab,
                                  sqlite3_index_info *pIdxInfo) {
  int hasSource = 0;
  int idxChunkSize = -1;
  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *pCons = &pIdxInfo->aConstraint[i];
    switch (pCons->iColumn) {
    case lembed_chunks_SOURCE: {
      if (!hasSource && !pCons->usable ||
          pCons->op != SQLITE_INDEX_CONSTRAINT_EQ)
        return SQLITE_CONSTRAINT;
      hasSource = 1;
      pIdxInfo->aConstraintUsage[i].argvIndex = 1;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      break;
    }
    case lembed_chunks_CHUNK_SIZE: {
    }
    }
  }
  if (!hasSource) {
    pVTab->zErrMsg = sqlite3_mprintf("source argument is required");
    return SQLITE_ERROR;
  }

  pIdxInfo->idxNum = 1;
  pIdxInfo->estimatedCost = (double)10;
  pIdxInfo->estimatedRows = 10;
  return SQLITE_OK;
}

static int lembed_chunksFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                               const char *idxStr, int argc,
                               sqlite3_value **argv) {
  lembed_chunks_cursor *pCur = (lembed_chunks_cursor *)pVtabCursor;
  struct Api *api = ((lembed_chunks_vtab *)pVtabCursor->pVtab)->api;
  struct llama_model *model;
  int rc = api_model_from_name(api, (const char *)sqlite3_value_text(argv[0]),
                               sqlite3_value_bytes(argv[0]), &model, NULL);
  pCur->iRowid = 0;

  char *input = (char *)sqlite3_value_text(argv[1]);
  sqlite3_int64 input_len = sqlite3_value_bytes(argv[1]);
  int32_t chunk_size = 5; // sqlite3_value_int(argv[1]);
  int32_t overlap = 0;    // argc > 2 ? sqlite3_value_int(argv[2]) : 0;

  int token_count;
  llama_token *tokens;
  rc = tokenize(model, input, input_len, &token_count, &tokens);
  assert(rc == SQLITE_OK);

  char *ptr = input;
  int nchunks = ceil(1.0 * token_count / chunk_size);
  pCur->chunks_count = nchunks;
  pCur->chunks = sqlite3_malloc(sizeof(char *) * nchunks);
  assert(pCur->chunks);

  for (int i = 0; i < nchunks; i++) {
    sqlite3_str *str_chunk = sqlite3_str_new(NULL);
    assert(str_chunk);

    for (int j = 0; j < chunk_size; j++) {
      int32_t token = tokens[i * chunk_size + j];
      int32_t piece_len_neg =
          llama_token_to_piece(model, token, NULL, 0, false);
      // printf("%d\n", piece_len_neg);
      // assert(piece_len_neg < 0);
      int32_t piece_len = abs(piece_len_neg);
      // include prefix space?
      // assert(piece_len > 1);
      if (!piece_len)
        continue;

      char *piece = sqlite3_malloc(piece_len);
      assert(piece);
      llama_token_to_piece(model, token, piece, piece_len, false);
      // printf("'%.*s' %d ", piece_len, piece, tokens[i*chunk_size + j]);

      char *begin = ptr;
      while (*ptr != piece[piece_len > 1 ? 1 : 0]) {
        ptr++;
      }
      sqlite3_str_append(str_chunk, begin, ptr - begin + piece_len);
      ptr += piece_len;

      sqlite3_free(piece);
    }

    char *chunk = sqlite3_str_finish(str_chunk);
    assert(chunk);
    pCur->chunks[i] = chunk;
  }

  return SQLITE_OK;
}

static int lembed_chunksRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  lembed_chunks_cursor *pCur = (lembed_chunks_cursor *)cur;
  *pRowid = pCur->iRowid;
  return SQLITE_OK;
}

static int lembed_chunksNext(sqlite3_vtab_cursor *cur) {
  lembed_chunks_cursor *pCur = (lembed_chunks_cursor *)cur;
  pCur->iRowid++;
  return SQLITE_OK;
}

static int lembed_chunksEof(sqlite3_vtab_cursor *cur) {
  lembed_chunks_cursor *pCur = (lembed_chunks_cursor *)cur;
  return pCur->iRowid >= pCur->chunks_count;
}

static int lembed_chunksColumn(sqlite3_vtab_cursor *cur,
                               sqlite3_context *context, int i) {
  lembed_chunks_cursor *pCur = (lembed_chunks_cursor *)cur;
  switch (i) {
  case lembed_chunks_CONTENTS:
    sqlite3_result_text(context, pCur->chunks[pCur->iRowid], -1, SQLITE_STATIC);
    break;
  case lembed_chunks_SOURCE:
    // TODO
    sqlite3_result_null(context);
    break;
  }
  return SQLITE_OK;
}

static sqlite3_module lembed_chunksModule = {
    /* iVersion    */ 0,
    /* xCreate     */ 0,
    /* xConnect    */ lembed_chunksConnect,
    /* xBestIndex  */ lembed_chunksBestIndex,
    /* xDisconnect */ lembed_chunksDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ lembed_chunksOpen,
    /* xClose      */ lembed_chunksClose,
    /* xFilter     */ lembed_chunksFilter,
    /* xNext       */ lembed_chunksNext,
    /* xEof        */ lembed_chunksEof,
    /* xColumn     */ lembed_chunksColumn,
    /* xRowid      */ lembed_chunksRowid,
    /* xUpdate     */ 0,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0};
#pragma endregion

#ifndef SQLITE_SUBTYPE
#define SQLITE_SUBTYPE 0x000100000
#endif

#ifndef SQLITE_RESULT_SUBTYPE
#define SQLITE_RESULT_SUBTYPE 0x001000000
#endif

#define SQLITE_LEMBED_DEBUG_STRING                                                \
  "Version: " SQLITE_LEMBED_VERSION "\n"                                          \
  "Date: " SQLITE_LEMBED_DATE "\n"                                                \
  "Commit: " SQLITE_LEMBED_SOURCE "\n"                                            \


#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_lembed_init(sqlite3 *db, char **pzErrMsg,
                            const sqlite3_api_routines *pApi) {
  SQLITE_EXTENSION_INIT2(pApi);

  llama_backend_init();
  llama_log_set(dummy_log, NULL);

  struct Api *a = sqlite3_malloc(sizeof(struct Api));
  assert(a);
  memset(a, 0, sizeof(*a));

  int rc = SQLITE_OK;
  const int DEFAULT_FLAGS =
      SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC;

  static const struct {
    char *zFName;
    void (*xFunc)(sqlite3_context *, int, sqlite3_value **);
    int nArg;
    int flags;
    void *p;
  } aFunc[] = {
      // clang-format off
    {"lembed_version", _static_text_func, 0, DEFAULT_FLAGS,  SQLITE_LEMBED_VERSION },
    {"lembed_debug",   _static_text_func, 0, DEFAULT_FLAGS,  SQLITE_LEMBED_DEBUG_STRING }
    // clang-format on
  };

  for (unsigned long i = 0;i < sizeof(aFunc) / sizeof(aFunc[0]) && rc == SQLITE_OK; i++) {
    rc = sqlite3_create_function_v2(db, aFunc[i].zFName, aFunc[i].nArg, aFunc[i].flags, aFunc[i].p, aFunc[i].xFunc, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("Error creating function %s: %s",
                                  aFunc[i].zFName, sqlite3_errmsg(db));
      return rc;
    }
  }

  static const struct {
    char *zFName;
    void (*xFunc)(sqlite3_context *, int, sqlite3_value **);
    int nArg;
  } aFuncApi[] = {
      // clang-format off
    {"lembed",                 lembed,                    2},
    {"lembed_tokenize_json",   lembed_tokenize_json,      2},
    {"lembed_token_score",     lembed_token_score,        2},
    {"lembed_token_to_piece",  lembed_token_to_piece_,    2},
    {"lembed_model_size",      lembed_model_size,         1},
    {"lembed_model_from_file", lembed_model_from_file,    1},
    {"lembed_model_options",   lembed_model_options_,     -1},
    {"lembed_context_options", lembed_context_options_,   -1},
    // clang-format on
  };
  for (unsigned long i = 0;i < sizeof(aFuncApi) / sizeof(aFuncApi[0]) && rc == SQLITE_OK; i++) {
    rc = sqlite3_create_function_v2(db, aFuncApi[i].zFName, aFuncApi[i].nArg, DEFAULT_FLAGS, a, aFuncApi[i].xFunc, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("Error creating function %s: %s",
                                  aFuncApi[i].zFName, sqlite3_errmsg(db));
      return rc;
    }
  }

  sqlite3_create_function_v2(db, "_lembed_api", 0, 0, a, _noop, NULL, NULL, api_free);

  sqlite3_create_module_v2(db, "lembed_chunks", &lembed_chunksModule, a, NULL);
  sqlite3_create_module_v2(db, "lembed_models", &lembed_modelsModule, a, NULL);
  return SQLITE_OK;
}
