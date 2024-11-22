cmake_minimum_required(VERSION 3.14)
project(SqliteEmbed C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(FetchContent)


# sqlite amalgamation, for up-to-date headers and sqlite3 CLI
set(SQLITE_VERSION 3470000)
set(SQLITE_YEAR 2024)
set(SQLITE_URL)
FetchContent_Declare(
  sqlite_amalgamation
  URL https://www.sqlite.org/${SQLITE_YEAR}/sqlite-amalgamation-${SQLITE_VERSION}.zip
  )
FetchContent_MakeAvailable(sqlite_amalgamation)


# llama.cpp
set(LLAMA_METAL OFF)
set(LLAMA_STATIC ON)
set(LLAMA_OPENMP OFF)

ExternalProject_Add(
  "llama.cpp"
  GIT_REPOSITORY git@github.com:ggerganov/llama.cpp.git
  GIT_TAG        b3091
)


# sqlite-lembed loadable
add_library(sqlite_lembed SHARED sqlite-lembed.c)
target_link_libraries(sqlite_lembed ggml_static llama)
target_include_directories(sqlite_lembed PRIVATE ${LLAMA_CPP_DIR})
target_include_directories(sqlite_lembed PRIVATE ${sqlite_amalgamation_SOURCE_DIR})
set_target_properties(sqlite_lembed PROPERTIES PREFIX "")
set_target_properties(sqlite_lembed PROPERTIES OUTPUT_NAME "lembed0")

# sqlite-lembed static
add_library(sqlite_lembed_static STATIC sqlite-lembed.c)
target_link_libraries(sqlite_lembed_static ggml_static llama)
target_include_directories(sqlite_lembed_static PRIVATE ${LLAMA_CPP_DIR})
target_include_directories(sqlite_lembed_static PRIVATE ${sqlite_amalgamation_SOURCE_DIR})
target_compile_definitions(sqlite_lembed_static PRIVATE SQLITE_CORE)
set_target_properties(sqlite_lembed_static PROPERTIES OUTPUT_NAME "sqlite_lembed0")


# sqlite-vec, for a better sqlite3 CLI
set(SQLITE_VEC_VERSION 0.1.6)
FetchContent_Declare(
  sqlite_vec
  URL https://github.com/asg017/sqlite-vec/releases/download/v${SQLITE_VEC_VERSION}/sqlite-vec-${SQLITE_VEC_VERSION}-amalgamation.tar.gz
)
FetchContent_MakeAvailable(sqlite_vec)


# sqlite3 CLI, with sqlite-lembed and sqlite-vec
add_executable(
  sqlite3_cli
  ${sqlite_amalgamation_SOURCE_DIR}/shell.c
  ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c
  ${sqlite_vec_SOURCE_DIR}/sqlite-vec.c
  core_init.c
)
add_dependencies(sqlite3_cli sqlite_lembed_static)
target_link_libraries(sqlite3_cli sqlite_lembed_static)
target_include_directories(
  sqlite3_cli PRIVATE
  ${sqlite_amalgamation_SOURCE_DIR}
  ${sqlite_vec_SOURCE_DIR}
)
target_compile_definitions(
  sqlite3_cli PUBLIC
  SQLITE_EXTRA_INIT=core_init
  SQLITE_CORE
)
set_target_properties(sqlite3_cli PROPERTIES OUTPUT_NAME "sqlite3")
