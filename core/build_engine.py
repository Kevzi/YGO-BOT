import os
import sys
import shutil
import subprocess
import urllib.request
import tarfile
from pathlib import Path

CORE_DIR = Path(__file__).parent.resolve()
LUA_DIR = CORE_DIR / "lua"
OCGCORE_DIR = CORE_DIR / "ocgcore_src"
BUILD_DIR = CORE_DIR / "build"

LUA_VERSION = "5.3.6"
LUA_URL = f"https://www.lua.org/ftp/lua-{LUA_VERSION}.tar.gz"

def setup_lua():
    if not (LUA_DIR / "lua.h").exists():
        print("Downloading and extracting Lua...")
        tar_path = CORE_DIR / f"lua-{LUA_VERSION}.tar.gz"
        if not tar_path.exists():
            urllib.request.urlretrieve(LUA_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=CORE_DIR)
        
        extracted = CORE_DIR / f"lua-{LUA_VERSION}"
        if LUA_DIR.exists():
            shutil.rmtree(LUA_DIR)
        extracted.rename(LUA_DIR)
        
        # move src contents to root of lua
        src_dir = LUA_DIR / "src"
        for f in src_dir.iterdir():
            target = LUA_DIR / f.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            f.rename(target)
    
    # Always ensure CMakeLists.txt exists
    cmake_content = """project(lua CXX)
file(GLOB LUA_SRC "*.c")
list(REMOVE_ITEM LUA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/lua.c" "${CMAKE_CURRENT_SOURCE_DIR}/luac.c")
set_source_files_properties(${LUA_SRC} PROPERTIES LANGUAGE CXX)
add_library(lua STATIC ${LUA_SRC})
"""
    with open(LUA_DIR / "CMakeLists.txt", "w", encoding="utf-8") as f:
        f.write(cmake_content)

def setup_master_cmake():
    cmake_content = """cmake_minimum_required(VERSION 3.10)
project(ygoenv CXX C)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lua)
add_subdirectory(ocgcore_src ocgcore)
"""
    with open(CORE_DIR / "CMakeLists.txt", "w", encoding="utf-8") as f:
        f.write(cmake_content)

def patch_ocgcore_cmake():
    cmake_path = OCGCORE_DIR / "CMakeLists.txt"
    # Overwrite ygopro-core CMakeLists.txt entirely to bypass custom macros
    content = """project(ocgcore CXX)
file(GLOB SRC "*.cpp")
add_library(ocgcore SHARED ${SRC})
target_include_directories(ocgcore PRIVATE ../lua)
target_link_libraries(ocgcore lua)
target_compile_definitions(ocgcore PRIVATE OCGCORE_EXPORT_FUNCTIONS)
"""
    with open(cmake_path, "w", encoding="utf-8") as f:
        f.write(content)

def build_engine():
    setup_lua()
    patch_ocgcore_cmake()
    setup_master_cmake()
    
    print("Running CMake configuration...")
    BUILD_DIR.mkdir(exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=BUILD_DIR, check=True)
    
    print("Building...")
    subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=BUILD_DIR, check=True)
    
    # Copy dll to core directory for easy ctypes loading
    if sys.platform.startswith("win"):
        ext = ".dll"
        possible_paths = [BUILD_DIR / "ocgcore" / "Release" / f"ocgcore{ext}", BUILD_DIR / "ocgcore" / f"ocgcore{ext}"]
    elif sys.platform.startswith("darwin"):
        ext = ".dylib"
        possible_paths = [BUILD_DIR / "ocgcore" / f"libocgcore{ext}", BUILD_DIR / "ocgcore" / "Release" / f"libocgcore{ext}"]
    else:
        ext = ".so"
        possible_paths = [BUILD_DIR / "ocgcore" / f"libocgcore{ext}", BUILD_DIR / "ocgcore" / "Release" / f"libocgcore{ext}"]
    
    dll_path = next((p for p in possible_paths if p.exists()), None)
    
    if dll_path:
        shutil.copy(dll_path, CORE_DIR / f"ocgcore{ext}")
        print(f"Engine library copied to core directory: {dll_path.name}")
    else:
        raise FileNotFoundError(f"Failed to find the compiled engine library (expected extension {ext}).")
    
    print("Build successful!")

if __name__ == "__main__":
    build_engine()
