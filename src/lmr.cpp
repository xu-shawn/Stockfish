#include "lmr.h"

#include <algorithm>
#include <cstdint>

namespace Stockfish {

namespace Search {

namespace LMR {

constexpr int input_weights[28][5] = {
  {52, -127, -22, -31, 31},   {-36, 78, -10, 14, 2},    {-94, 3, 13, 47, 123},
  {190, -33, -41, 10, -7},    {-1, -122, 77, 88, -19},  {122, -32, -15, -84, -8},
  {155, -52, -2, -99, 34},    {-6, -15, 27, 78, -116},  {-115, -24, -29, -15, -97},
  {1, -69, -20, -7, -72},     {60, -72, -79, 23, -92},  {91, 2, -31, -14, -167},
  {24, -14, -118, -65, 19},   {15, -29, 183, -29, -50}, {31, -87, -39, -113, -127},
  {36, 83, -3, 44, 15},       {113, -19, -75, 80, 64},  {-66, -17, -21, -4, 16},
  {63, 26, -58, 48, 69},      {-96, 32, 56, -80, -32},  {105, 73, 104, -123, -48},
  {117, -5, -6, -3, -6},      {87, 57, -71, 95, 43},    {12, 78, -28, 79, 66},
  {-113, -10, -212, 77, 163}, {0, 109, -13, 163, -15},  {-68, 5, 95, 6, -27},
  {81, 12, 122, 23, 7}};
constexpr int output_weights[28] = {-67, 69,   46,  -14, 31,  -1,  -97, 0,   -77, 134,
                                    -44, -19,  149, -95, -88, 10,  -42, 132, -48, 68,
                                    42,  -152, 38,  9,   33,  -67, 102, -70};
constexpr int biases[28] = {25,  -31, -65, 17, -225, -13, 59,  -52,  43,  185, 40, 131,  -73, -94,
                            -46, -30, -39, 15, -33,  38,  -80, -178, -51, 4,   51, -156, 83,  111};

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

    return reduction / 1024;
}

}  // namespace LMR

}  // namespace Search

}  // namespace Stockfish
