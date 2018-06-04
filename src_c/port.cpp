#include <iostream>
#include <stdint.h>
#include "sha256.h"

extern "C" BYTE * start_mine(BYTE bhash[32], BYTE nonce[32], WORD bdiff, WORD sdiff, WORD id);

int is_big_endian(void)
{
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}

uint32_t swapByteOrder(uint32_t ui)
{
    ui = (ui >> 24) |
         ((ui<<8) & 0x00FF0000) |
         ((ui>>8) & 0x0000FF00) |
         (ui << 24);
    return ui;
}

uint32_t read_32b_integer(std::istream& s)
{
    uint32_t value;
    s.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is_big_endian())
        value = swapByteOrder(value);
//    std::cerr << 'v' << value << "\n\r";
//    std::cerr.flush();
    return value;
}

std::ostream&  write_32b_integer(std::ostream& s, uint32_t len)
{
//    std::cerr << 'w' << len << "\n\r";
//    std::cerr.flush();

    if (!is_big_endian())
        len = swapByteOrder(len);
    s.write(reinterpret_cast<char*>(&len), sizeof(len));
    return s;
}

int main(void)
{
    // Helps to detect std::cin closing
    std::cin.exceptions ( std::istream::failbit | std::istream::badbit );
    try {

    // Read messages from Erlang port
    while(true) {
        // read packet length, 4 bytes
        uint32_t len = read_32b_integer(std::cin);

        // read data, 32 bytes
        BYTE* bhash = new BYTE[32];
        std::cin.read((char*)bhash, 32);

        // read data, 32 bytes
        BYTE* nonce = new BYTE[32];
        std::cin.read((char*)nonce, 32);

        // read difficulty, 4 bytes
        uint32_t block_diff = read_32b_integer(std::cin);

        // read difficulty, 4 bytes
        uint32_t share_diff = read_32b_integer(std::cin);

        // read core_id, 4 bytes
        uint32_t id = read_32b_integer(std::cin);

        start_mine(bhash, nonce, block_diff, share_diff, id);

        write_32b_integer(std::cout, 32);
//        std::cout.write(hash, 32);
        std::cout.write((char*)nonce, 32);
//        write_32b_integer(std::cout, diff);
//        write_32b_integer(std::cout, id);
        std::cout.flush();

        delete[] bhash;
        delete[] nonce;
    }
    } catch ( std::istream::failure e1 ) {
      std::cerr << "std::cin closed\n\r";
      std::cerr.flush();
      return 0;
    } catch ( std::exception e2 ) {
      return 0;
    }
    return 0;
}
