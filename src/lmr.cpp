#include "lmr.h"

#include <algorithm>
#include <cstdint>

#include "tune.h"

namespace Stockfish {

namespace Search {

namespace LMR {

constexpr int input_weights[28][5] = {
  {-8, -3, 15, -8, 5},   {30, 10, -2, -2, -4}, {11, -18, 0, 7, -2},   {1, -24, 1, 11, 1},
  {12, -14, 2, -4, 10},  {-14, 4, -6, 16, 2},  {-10, -1, 9, 13, 2},   {-10, -1, 2, -3, -23},
  {-15, -1, 8, 0, 4},    {7, -2, 4, -4, 7},    {-5, -5, -7, 10, -5},  {9, -7, -10, 2, 2},
  {-6, 2, -5, 1, -7},    {15, 2, 0, 7, 2},     {-13, -14, 8, 10, 9},  {-2, 21, 4, -5, 9},
  {1, -4, 18, -7, -18},  {-7, 1, 0, 5, -17},   {-13, 6, 0, -15, -8},  {-9, -3, -8, 10, -10},
  {3, -21, 11, 16, -15}, {-6, 16, 11, 9, -4},  {-14, -21, 1, 2, -10}, {17, 0, 6, 3, 6},
  {-9, 7, -19, 6, -2},   {6, -4, 8, -13, -2},  {-4, 7, 1, -3, -5},    {7, 8, 2, 12, -1}};
constexpr int output_weights[28] = {-3, 18, 10,  -9,  2,  1,  0, -1, -10, 5, -11, 12, -11, 8,
                                    -1, 10, -14, -15, -1, -9, 0, -6, -3,  7, 12,  8,  -14, -1};
constexpr int biases[28]         = {-7,  4,  7,   -17, -1, 3,  -5,  3,  3, 7,  -12, -6, -12, 2,
                                    -15, -4, -16, -4,  0,  -1, -15, -4, 3, 18, 11,  -3, 4,   -1};

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
            reduction += output_weights[i] * std::clamp<int32_t>(biases[i], -64, 64);

        else
        {
            int32_t value = biases[i];

            for (int j = 0; j < 5; j++)
                value += input_weights[i][j] * data[j];

            reduction += output_weights[i] * std::clamp(value, -64, 64);
        }
    }

    return reduction / 64;
}

}  // namespace LMR

}  // namespace Search

}  // namespace Stockfish
