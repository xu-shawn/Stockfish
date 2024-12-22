/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LMR_H_INCLUDED
#define LMR_H_INCLUDED

#include <cstdint>

namespace Stockfish {

namespace Search {

namespace LMR {

class Network {
   public:
    Network() = default;
    void init_node(const bool data[8]);
    int  get_reduction(const int32_t data[5]) const;

   private:
    /* int8_t  input_weights[28][5]; */
    /* int8_t  output_weights[28]; */
    /* int8_t biases[28]; */

    // Calculated at runtime
    bool enabled[28];
};

}  // namespace LMR

}  // namespace Search

}  // namespace Stockfish

#endif
