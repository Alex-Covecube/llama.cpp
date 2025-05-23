message(STATUS "Using NUMA backend")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

ggml_add_backend_library(ggml-numa
                         ggml-numa.cpp
                        )

target_link_libraries(ggml-numa PRIVATE numa)
