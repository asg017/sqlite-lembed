# ruff: noqa: E731
import struct
import re
import pytest
import sqlite3
import inspect
from contextlib import contextmanager

EXT_PATH = "./dist/lembed0"
MODEL1_PATH = "./dist/.models/all-MiniLM-L6-v2.e4ce9877.q8_0.gguf"
MODEL2_PATH = "./dist/.models/mxbai-embed-xsmall-v1-q8_0.gguf"
MODEL3_PATH = "./dist/.models/nomic-embed-text-v1.5.Q2_K.gguf"


def connect(ext, path=":memory:", extra_entrypoint=None):
    db = sqlite3.connect(path)

    db.execute(
        "create temp table base_functions as select name from pragma_function_list"
    )
    db.execute("create temp table base_modules as select name from pragma_module_list")

    db.enable_load_extension(True)
    db.load_extension(ext)

    if extra_entrypoint:
        db.execute("select load_extension(?, ?)", [ext, extra_entrypoint])

    db.execute(
        "create temp table loaded_functions as select name from pragma_function_list where name not in (select name from base_functions) order by name"
    )
    db.execute(
        "create temp table loaded_modules as select name from pragma_module_list where name not in (select name from base_modules) order by name"
    )

    db.row_factory = sqlite3.Row
    return db


db = connect(EXT_PATH)


def explain_query_plan(sql):
    return db.execute("explain query plan " + sql).fetchone()["detail"]


def execute_all(cursor, sql, args=None):
    if args is None:
        args = []
    results = cursor.execute(sql, args).fetchall()
    return list(map(lambda x: dict(x), results))


def spread_args(args):
    return ",".join(["?"] * len(args))


@contextmanager
def _raises(message, error=sqlite3.OperationalError):
    with pytest.raises(error, match=re.escape(message)):
        yield


FUNCTIONS = [
    "_lembed_api",
    "lembed",
    "lembed",
    "lembed_context_options",
    "lembed_debug",
    "lembed_model_from_file",
    "lembed_model_options",
    "lembed_model_size",
    "lembed_token_score",
    "lembed_token_to_piece",
    "lembed_tokenize_json",
    "lembed_tokenize_json",
    "lembed_version",
]
MODULES = [
    "lembed_batch",
    "lembed_models",
]


def test_funcs():
    funcs = list(
        map(
            lambda a: a[0],
            db.execute("select name from loaded_functions").fetchall(),
        )
    )
    assert funcs == FUNCTIONS


def test_modules():
    modules = list(
        map(lambda a: a[0], db.execute("select name from loaded_modules").fetchall())
    )
    assert modules == MODULES


def test_lembed_version():
    lembed_version = lambda *args: db.execute(
        "select lembed_version()", args
    ).fetchone()[0]
    assert lembed_version()[0] == "v"


def test_lembed_debug():
    lembed_debug = lambda *args: db.execute("select lembed_debug()", args).fetchone()[0]
    d = lembed_debug().split("\n")
    assert len(d) == 4


def test_lembed():
    lembed = lambda *args: db.execute(
        "select lembed({})".format(spread_args(args)), args
    ).fetchone()[0]

    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["aaa", MODEL1_PATH],
    )
    a = lembed("aaa", "alex garcia")
    assert len(a) == (384 * 4)
    assert struct.unpack("1f", a[0:4])[0] == pytest.approx(
        -0.09205757826566696, rel=1e-2
    )

    with _raises(
        "Unknown model name 'aaaaaaaaa'. Was it registered with lembed_models?"
    ):
        lembed("aaaaaaaaa", "alex garcia")

def test_lembed_multiple():
    db = connect(EXT_PATH)
    lembed = lambda *args: db.execute(
        "select lembed({})".format(spread_args(args)), args
    ).fetchone()[0]

    db.execute(
        "insert into temp.lembed_models(name, model) values (?, ?), (?, ?), (?, ?)",
        ["aaa", MODEL1_PATH, "bbb", MODEL2_PATH, "ccc", MODEL3_PATH],
    )
    a = lembed("aaa", "alex garcia")
    b = lembed("bbb", "alex garcia")
    c = lembed("ccc", "alex garcia")
    assert len(a) == (384 * 4)
    assert len(b) == (384 * 4)
    assert len(c) == (768 * 4)

    assert execute_all(db, "select * from lembed_models") == [
        {"name": "aaa", "model": None, "dimensions": 384, "n_ctx": 512, "pooling_type": "none"},
        {"name": "bbb", "model": None, "dimensions": 384, "n_ctx": 512, "pooling_type": "mean"},
        {"name": "ccc", "model": None, "dimensions": 768, "n_ctx": 512, "pooling_type": "mean"},
    ]


def test_lembed_default():
    db = connect(EXT_PATH)
    lembed = lambda *args: db.execute(
        "select lembed({})".format(spread_args(args)), args
    ).fetchone()[0]

    with _raises("No default model has been registered yet with lembed_models"):
        lembed("alex garcia")

    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["default", MODEL1_PATH],
    )
    a = lembed("alex garcia")
    assert len(a) == (384 * 4)
    assert struct.unpack("1f", a[0:4])[0] == pytest.approx(
        -0.09205757826566696, rel=1e-2
    )

    # test 2: try with NULL name
    db = connect(EXT_PATH)
    lembed = lambda *args: db.execute(
        "select lembed({})".format(spread_args(args)), args
    ).fetchone()[0]

    with _raises("No default model has been registered yet with lembed_models"):
        lembed("alex garcia")

    db.execute(
        "insert into temp.lembed_models(model) values (lembed_model_from_file(?))",
        [MODEL1_PATH],
    )
    a = lembed("alex garcia")
    assert len(a) == (384 * 4)
    assert struct.unpack("1f", a[0:4])[0] == pytest.approx(
        -0.09205757826566696, rel=1e-2
    )

    # test 3: try text path to model
    db = connect(EXT_PATH)
    lembed = lambda *args: db.execute(
        "select lembed({})".format(spread_args(args)), args
    ).fetchone()[0]

    with _raises("No default model has been registered yet with lembed_models"):
        lembed("alex garcia")

    db.execute(
        "insert into temp.lembed_models(model) values (?)",
        [MODEL1_PATH],
    )
    a = lembed("alex garcia")
    assert len(a) == (384 * 4)
    assert struct.unpack("1f", a[0:4])[0] == pytest.approx(
        -0.09205757826566696, rel=1e-2
    )

def test_stress_mxbai_xsmall():
    db = connect(EXT_PATH)
    lembed = lambda *args: db.execute(
        "select lembed({})".format(spread_args(args)), args
    ).fetchone()[0]

    with _raises("No default model has been registered yet with lembed_models"):
        lembed("alex garcia")

    db.execute(
        "insert into temp.lembed_models(name, model) values (?, lembed_model_from_file(?))",
        ["default", MODEL1_PATH],
    )
    assert len(lembed("a " * 256)) == 384*4
    #print(db.execute('select lembed_tokenize_json(\'a a a a\') as x').fetchone()["x"])

    # including start and end token, this is 512 tokens, max ctx size for all-mini
    lembed("a " * (510))

    with _raises("Error generating embedding: Input too long, provided 513 tokens, but model has context size of 512"):
      lembed("a " * (511))

    with _raises("Error generating embedding: Input too long, provided 4098 tokens, but model has context size of 512"):
      lembed("a " * (4096))

@pytest.mark.skip(reason="TODO")
def test__lembed_api():
    _lembed_api = lambda *args: db.execute("select _lembed_api()", args).fetchone()[0]
    pass


@pytest.mark.skip(reason="TODO")
def test_lembed_context_options():
    lembed_context_options = lambda *args: db.execute(
        "select lembed_context_options()", args
    ).fetchone()[0]
    pass


@pytest.mark.skip(reason="TODO")
def test_lembed_model_size():
    lembed_model_size = lambda *args: db.execute(
        "select lembed_model_size()", args
    ).fetchone()[0]
    pass


@pytest.mark.skip(reason="TODO")
def test_lembed_model_from_file():
    lembed_model_from_file = lambda *args: db.execute(
        "select lembed_model_from_file()", args
    ).fetchone()[0]
    pass


@pytest.mark.skip(reason="TODO")
def test_lembed_model_options():
    lembed_model_options = lambda *args: db.execute(
        "select lembed_model_options()", args
    ).fetchone()[0]
    pass


@pytest.mark.skip(reason="TODO")
def test_lembed_tokenize_json():
    lembed_tokenize_json = lambda *args: db.execute(
        "select lembed_tokenize_json()", args
    ).fetchone()[0]
    pass


@pytest.mark.skip(reason="TODO")
def test_lembed_token_score():
    lembed_token_score = lambda *args: db.execute(
        "select lembed_token_score()", args
    ).fetchone()[0]
    pass


@pytest.mark.skip(reason="TODO")
def test_lembed_token_to_piece():
    lembed_token_to_piece = lambda *args: db.execute(
        "select lembed_token_to_piece()", args
    ).fetchone()[0]
    pass

@pytest.mark.skip(reason="TODO")
def test_lembed_batch():
    lembed_batch = lambda *args: db.execute(
        "select * from lembed_batch()", args
    ).fetchone()[0]
    pass


@pytest.mark.skip(reason="TODO")
def test_lembed_models():
    lembed_models = lambda *args: db.execute(
        "select * from lembed_chunks()", args
    ).fetchone()[0]
    pass


def test_coverage():
    current_module = inspect.getmodule(inspect.currentframe())
    test_methods = [
        member[0]
        for member in inspect.getmembers(current_module)
        if member[0].startswith("test_")
    ]
    funcs_with_tests = set([x.replace("test_", "") for x in test_methods])
    for func in [*FUNCTIONS, *MODULES]:
        assert func in funcs_with_tests, f"{func} is not tested"
