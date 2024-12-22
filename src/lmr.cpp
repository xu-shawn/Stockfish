#include "lmr.h"

#include <algorithm>
#include <cstdint>

#include "tune.h"

namespace Stockfish {

namespace Search {

namespace LMR {

int input_weights[28][5] = {{0}};
int output_weights[28]   = {0};
int biases[28]           = {0};

TUNE(SetRange(-256, 256), input_weights);
TUNE(SetRange(-256, 256), output_weights);
TUNE(SetRange(-256, 256), biases);

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
