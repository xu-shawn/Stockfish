#include "lmr.h"

#include <algorithm>
#include <cstdint>

#include "misc.h"
#include "tune.h"

namespace Stockfish {

namespace Search {

namespace LMR {

constexpr int input_weights[28][5] = {
  {-314, 125, 31, 157, -10},    {-137, 6, 122, 158, 50},      {-399, 57, 127, 15, -69},
  {-175, 105, 68, -219, -30},   {-15, 140, 333, -309, -101},  {-17, -250, -158, -60, 62},
  {155, -105, 223, -71, 38},    {-70, -71, -165, 302, 43},    {-384, 240, -216, 3, -70},
  {197, -284, -214, 46, -383},  {2, 211, -51, -185, 57},      {175, -172, 18, 244, -477},
  {-343, -113, 284, -145, 331}, {-123, -66, 36, 202, -176},   {-149, 147, 416, 151, -464},
  {66, -225, -232, -191, -346}, {260, 245, -26, -163, -419},  {109, 288, -256, -191, 249},
  {-168, 9, 94, 149, -52},      {-115, 249, 18, -179, -276},  {39, 73, -378, -481, 45},
  {162, -104, -73, -184, -151}, {102, 229, 63, 48, -127},     {-39, -238, 120, 67, -326},
  {154, 115, -229, 574, 156},   {-159, -273, -466, -23, 178}, {-145, 40, -246, -72, -76},
  {-40, 292, 228, 174, -163}};
constexpr int output_weights[28] = {52,   219, -268, 43,  89,  -145, 198,  -61,  107,  -231,
                                    -120, 27,  475,  132, 409, 40,   -227, -110, -477, 66,
                                    203,  65,  119,  48,  48,  -257, 4,    -239};
constexpr int biases[28]         = {-142, 185,  58,   93,   -88, -371, 35,   -154, -52, -87,
                                    114,  115,  -201, -202, 20,  160,  -69,  73,   72,  132,
                                    -7,   -298, -360, 3,    -84, -157, -180, -41};

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

    reduction = reduction / 1024;

    return reduction;
}

}  // namespace LMR

}  // namespace Search

}  // namespace Stockfish
