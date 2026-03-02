cmake_minimum_required(VERSION 3.14)
project(flappyBird CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        master
)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

add_executable(flappy main.cpp)

target_link_libraries(flappy PRIVATE raylib)

if(APPLE)
    target_link_libraries(flappy PRIVATE "-framework IOKit")
    target_link_libraries(flappy PRIVATE "-framework Cocoa")
    target_link_libraries(flappy PRIVATE "-framework OpenGL")
endif()
