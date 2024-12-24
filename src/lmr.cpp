#include "lmr.h"

#include <algorithm>
#include <cstdint>

#include "tune.h"

namespace Stockfish {

namespace Search {

namespace LMR {

constexpr int input_weights[28][5] = {
  {383, -86, -355, 5, -131},     {-360, -102, 465, -238, -505}, {-82, 433, 872, 120, -490},
  {208, 237, 167, 49, -221},     {363, -26, -159, 505, -229},   {-74, -335, -321, 246, -25},
  {73, -499, 622, -812, -383},   {398, 310, 38, -46, 73},       {-367, -301, -161, 803, -967},
  {-154, -514, 267, -128, -261}, {-673, -3, -59, -763, 162},    {-664, 150, 446, 453, -365},
  {92, 343, 412, -333, -623},    {-408, -357, 275, -694, 473},  {-606, 182, 418, 205, 289},
  {35, -102, -333, -271, -290},  {240, -170, 162, 397, -258},   {429, 229, 137, -117, -504},
  {-91, -53, 955, 1139, 161},    {239, -549, 240, -343, -748},  {798, -274, 744, -189, 434},
  {479, 103, -183, 87, -240},    {492, 476, 439, 629, 381},     {129, 188, 345, -393, -109},
  {-516, -13, 552, -665, 692},   {-683, 9, -48, -484, 571},     {-562, 489, -375, -142, 194},
  {469, -387, -242, -754, -132}};
constexpr int output_weights[28] = {184,  531,  485, -364, -909, -506, 42,   -10,  212, 299,
                                    -503, 21,   -96, -678, -481, -49,  246,  -217, 98,  -67,
                                    -118, -151, 951, 470,  204,  -673, -456, -457};
constexpr int biases[28]         = {329, -472, 223,  -553, -144, -95,  171,  -243, 375, -14,
                                    45,  332,  139,  -247, -49,  -48,  -421, -681, 173, 317,
                                    63,  -88,  -445, 331,  257,  -524, 192,  93};

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

    return reduction / 64;
}

}  // namespace LMR

}  // namespace Search

}  // namespace Stockfish
