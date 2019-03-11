cmake_minimum_required(VERSION 3.13)
project(XOR_nn C)

set(CMAKE_C_STANDARD 11)

add_executable(XOR_nn main.c)

target_link_libraries(XOR_nn m)