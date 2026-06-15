cmake_minimum_required(VERSION 3.10)
project(ygoenv CXX C)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lua)
add_subdirectory(ocgcore_src ocgcore)
