#include <iostream>
#include <stdint.h>
struct loc_info {
    uint8_t controler;
    uint8_t location;
    uint32_t sequence;
    uint32_t position;
};
int main() { std::cout << sizeof(loc_info) << std::endl; return 0; }
