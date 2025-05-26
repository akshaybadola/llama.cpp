# gemma3 (custom)

add_library(gemma3 SHARED gemma3_c_api.cpp gemma3.h)
target_link_libraries(gemma3 PRIVATE llama common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(gemma3 PRIVATE ../../common  ../../tools)
