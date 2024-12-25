#include "lmr.h"
#include "tune.h"

#include <algorithm>
#include <cstdint>

namespace Stockfish {

namespace Search {

namespace LMR {

int input_weights[28][5] = {
  {-314, 171, 106, 206, -43},   {-168, 67, 232, 145, -43},    {-270, 64, 144, -64, -2},
  {-176, -80, 98, -233, -83},   {-74, 77, 214, -332, -243},   {-34, -382, -169, -17, 107},
  {353, -71, 321, -132, 127},   {-27, 49, -149, 320, -85},    {-376, 141, -342, -73, -83},
  {174, -333, -149, -21, -259}, {89, 95, 32, -339, 155},      {315, -122, -125, 153, -522},
  {-330, -177, 207, -181, 380}, {-249, -9, -47, 298, -129},   {-90, 97, 473, 115, -529},
  {130, -194, -216, -57, -217}, {290, 370, -118, -172, -427}, {185, 297, -235, -195, 311},
  {-247, -16, 54, 334, -115},   {-67, 224, -42, -259, -152},  {-94, 224, -231, -502, -56},
  {137, -80, -255, -253, -25},  {147, 353, 192, -12, 16},     {-221, -243, 211, -9, -315},
  {180, 205, -174, 553, 61},    {-100, -294, -496, -47, 16},  {-169, -77, -375, -5, -58},
  {-31, 193, 330, 171, -202}};
int output_weights[28] = {-130, 171, -210, -88, -11, -81,  92,   -119, 19,   -266,
                          39,   22,  453,  27,  414, -102, -301, -42,  -444, -69,
                          176,  126, 321,  -29, 48,  -217, 101,  -145};
int biases[28] = {-142, 249, -31, 87, -46, -226, -160, -148, -92,  38,   15,  -37,  -209, -277,
                  68,   241, -7,  44, -82, 73,   -90,  -189, -437, -176, -50, -151, -181, -197};

TUNE(SetRange(-1024, 1024), input_weights);
TUNE(SetRange(-512, 512), output_weights);
TUNE(SetRange(-512, 512), biases);

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
