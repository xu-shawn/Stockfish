#include "lmr.h"
#include "tune.h"

#include <algorithm>
#include <cstdint>

namespace Stockfish {

namespace Search {

namespace LMR {

constexpr std::int16_t input_weights[28][5] = {
  {-448, 179, -2, 155, -70},    {-189, 152, 134, 59, -128},   {-324, 179, 105, -202, 7},
  {-103, 59, 70, -276, -77},    {-197, 150, 380, -285, -148}, {-174, -444, -177, 20, 259},
  {399, -24, 251, -162, 33},    {60, -55, -153, 290, -131},   {-364, 90, -375, -138, -132},
  {219, -411, -51, -34, -212},  {42, 144, 13, -435, 12},      {306, -147, -49, 246, -372},
  {-335, -159, 237, -164, 364}, {-187, -149, -26, 212, -134}, {-195, 108, 663, 138, -552},
  {62, -91, -358, -140, -313},  {206, 317, -112, -191, -510}, {286, 326, -237, -198, 220},
  {-299, 23, 29, 242, -166},    {10, 227, -3, -182, -181},    {-61, 286, -165, -487, -59},
  {170, -66, -358, -276, -37},  {252, 500, 110, -145, 2},     {-230, -185, 64, 20, -331},
  {148, 209, -188, 573, 94},    {-158, -265, -584, -22, -14}, {-164, 45, -384, -20, -160},
  {-35, 227, 355, 136, -31}};
constexpr std::int16_t output_weights[28] = {
  -116, 195,  -225, -89,  -88,  -59, 57,  -136, 50,  -231, 28, 26,   419, 15,
  425,  -141, -286, -100, -438, -63, 150, 165,  304, -98,  50, -233, 90,  -73};
constexpr std::int16_t biases[28] = {-106, 203,  -65,  110,  -100, -281, -161, -135, -77,  77,
                                     72,   41,   -141, -204, 109,  172,  16,   21,   -118, 53,
                                     -70,  -140, -413, -230, -44,  -103, -230, -145};

void Network::init_node(const bool data[8]) {
    int counter = 0;

    for (int i = 0; i < 8; i++)
        for (int j = i + 1; j < 8; j++)
            enabled[counter++] = data[i] ^ data[j];
}

int Network::get_reduction(const int32_t data[5]) const {
    int32_t reduction = 0;

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
