
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cu include/*.cuh)


set(TARGET_NAME program_main)

add_executable(${TARGET_NAME} ${srcs})

# 为 program_main 添加头文件搜索路径
target_include_directories(program_main PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PUBLIC
    util
)