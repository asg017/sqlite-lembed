cmake_minimum_required(VERSION 3.14)
project(SqliteEmbed C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(LLAMA_METAL OFF)
set(LLAMA_STATIC ON)
set(LLAMA_OPENMP OFF)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")
set(SQLITE_AMALGAMATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/sqlite")

include(ExternalProject)

set(SQLITE_VERSION 3450300)
set(SQLITE_YEAR 2024)
set(SQLITE_URL https://www.sqlite.org/${SQLITE_YEAR}/sqlite-amalgamation-${SQLITE_VERSION}.zip)
ExternalProject_Add(sqlite_amalgamation
    URL             ${SQLITE_URL}
    DOWNLOAD_DIR    ${CMAKE_BINARY_DIR}/downloads
    SOURCE_DIR      ${SQLITE_AMALGAMATION_DIR}
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     ""
)


add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)

include_directories(${SQLITE_AMALGAMATION_DIR})

add_library(sqlite_lembed SHARED sqlite-lembed.c)
add_dependencies(sqlite_lembed sqlite_amalgamation)
target_link_libraries(sqlite_lembed ggml_static llama)
target_include_directories(sqlite_lembed PRIVATE ${LLAMA_CPP_DIR})
set_target_properties(sqlite_lembed PROPERTIES PREFIX "")
set_target_properties(sqlite_lembed PROPERTIES OUTPUT_NAME "lembed0")

add_library(sqlite_lembed_static STATIC sqlite-lembed.c)
add_dependencies(sqlite_lembed_static sqlite_amalgamation)
target_link_libraries(sqlite_lembed_static ggml_static llama)
target_include_directories(sqlite_lembed_static PRIVATE ${LLAMA_CPP_DIR})
target_compile_definitions(sqlite_lembed_static PRIVATE SQLITE_CORE)
set_target_properties(sqlite_lembed_static PROPERTIES OUTPUT_NAME "sqlite_lembed0")
