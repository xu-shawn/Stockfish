#include "lmr.h"

#include <algorithm>
#include <cstdint>

#include "tune.h"

namespace Stockfish {

namespace Search {

namespace LMR {

constexpr int input_weights[28][5] = {
  {157, -100, -60, -28, 7},   {-13, 149, 202, 57, -64},   {-159, -66, 125, -22, 171},
  {127, -7, 144, -145, 86},   {97, 33, 1, -109, 104},     {123, 64, 24, -104, -26},
  {74, 42, 32, -62, 126},     {-25, 158, 54, -154, -220}, {-47, 15, -87, 231, -125},
  {-138, -109, -88, 30, 168}, {17, 200, -142, 33, 104},   {64, 110, -114, 24, -42},
  {-1, 3, -40, 1, -141},      {114, 68, 108, 210, 1},     {-4, -146, -27, -25, -205},
  {-104, -17, 47, 40, -3},    {-226, -58, -8, 80, -44},   {-93, 76, 152, -19, -5},
  {-197, 262, 78, -6, -73},   {-167, 73, -115, 56, 30},   {32, 25, 145, 79, -103},
  {-95, -26, -8, -123, -52},  {280, -163, 99, -7, -63},   {-21, 68, -10, -75, -57},
  {-175, 185, 49, 22, 144},   {223, -65, 127, -27, 4},    {-68, -139, 148, 109, 21},
  {103, 103, 205, 59, 136}};
constexpr int output_weights[28] = {-1, -1, 0, 1, 1, 0,  0, 1, -5, 2, 0,  2, -1, -2,
                                    -2, -2, 2, 2, 0, -3, 0, 1, -1, 2, -1, 1, -1, 0};
constexpr int biases[28]         = {-2,   65,  -43,  88,  126, 11,   -118, -9, -6,  -99,
                                    -141, -84, 121,  -74, 83,  -118, -163, 38, -42, 102,
                                    -105, -62, -117, 127, 7,   -14,  104,  -11};

void Network::init_node(const bool data[8]) {
    int counter = 0;

    for (int i = 0; i < 8; i++)
        for (int j = i + 1; j < 8; j++)
            enabled[counter++] = data[i] ^ data[j];
}

int Network::get_reduction(const int32_t data[5]) const {
    int reduction = 0;

    for (int i = 0; i < 28; i++)
    {
        if (!enabled[i])
            reduction += output_weights[i] * std::clamp<int32_t>(biases[i], 0, 1024);

        else
        {
            int32_t value = biases[i];

            for (int j = 0; j < 5; j++)
                value += input_weights[i][j] * data[j];

            reduction += output_weights[i] * std::clamp(value, 0, 1024);
        }
    }

    return reduction / 128;
}

}  // namespace LMR

}  // namespace Search

}  // namespace Stockfish
