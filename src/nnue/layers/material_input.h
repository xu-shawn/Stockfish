/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

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

// Definition of layer MaterialInput of NNUE evaluation function

#ifndef NNUE_LAYERS_MATERIAL_INPUT_H_INCLUDED
#define NNUE_LAYERS_MATERIAL_INPUT_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <iostream>

#include "../../bitboard.h"
#include "../../position.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

// One-hot features of the material configuration seen from the side to move, whose
// weighted sum is added to the outputs of the first hidden layer of every layer stack.
// With P, N, B, R, Q the piece counts:
//   - per side (side to move, then opponent): min(count, cap) of P, N, B, R, Q with caps
//     8, 2, 2, 2, 1 (40 features)
//   - side to move minus opponent: N + B clamped to [-3, 3], R to [-2, 2], Q to [-1, 1],
//     P to [-4, 4] (24 features)
//   - bishop pair (bishops on both square colours), per side (2 features)
//   - joint non-pawn signature: per side s = min(Q, 1) * 27 + min(R, 2) * 9 + min(B, 2) * 3
//     + min(N, 2), feature s(side to move) * 54 + s(opponent) (2916 features)
// The weights are i32 in the scale of the first layer's accumulators.
template<IndexType OutDims>
class MaterialInput {
   public:
    using OutputType = i32;

    static constexpr IndexType InputDimensions  = 2982;
    static constexpr IndexType OutputDimensions = OutDims;
    static constexpr IndexType MaxActive        = 17;

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() { return 0x4D415431u; }  // "MAT1"

    static int active_features(const Position& pos, IndexType* out) {
        constexpr int       Caps[5]  = {8, 2, 2, 2, 1};
        constexpr PieceType Types[5] = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN};
        constexpr Bitboard  Light    = 0x55AA55AA55AA55AAULL;

        const Color sides[2] = {pos.side_to_move(), ~pos.side_to_move()};
        int         count[2][5];
        for (int s = 0; s < 2; ++s)
            for (int t = 0; t < 5; ++t)
                count[s][t] = popcount(pos.pieces(sides[s], Types[t]));

        int n = 0, offset = 0;
        for (int s = 0; s < 2; ++s)
            for (int t = 0; t < 5; ++t)
            {
                out[n++] = offset + std::min(count[s][t], Caps[t]);
                offset += Caps[t] + 1;
            }

        const auto diff = [&](int d, int limit) {
            out[n++] = offset + std::clamp(d, -limit, limit) + limit;
            offset += 2 * limit + 1;
        };
        diff(count[0][1] + count[0][2] - count[1][1] - count[1][2], 3);
        diff(count[0][3] - count[1][3], 2);
        diff(count[0][4] - count[1][4], 1);
        diff(count[0][0] - count[1][0], 4);

        for (int s = 0; s < 2; ++s)
        {
            const Bitboard bishops = pos.pieces(sides[s], BISHOP);
            if ((bishops & Light) && (bishops & ~Light))
                out[n++] = offset + s;
        }
        offset += 2;

        const auto signature = [&](int s) {
            return std::min(count[s][4], 1) * 27 + std::min(count[s][3], 2) * 9
                 + std::min(count[s][2], 2) * 3 + std::min(count[s][1], 2);
        };
        out[n++] = offset + signature(0) * 54 + signature(1);
        offset += 54 * 54;

        assert(offset == int(InputDimensions) && n <= int(MaxActive));
        return n;
    }

    // Writes biases plus the weights of the active features to output
    void propagate(const Position& pos, OutputType* output) const {
        IndexType active[MaxActive];
        const int n = active_features(pos, active);

        for (IndexType o = 0; o < OutputDimensions; ++o)
            output[o] = biases[o];
        for (int k = 0; k < n; ++k)
        {
            const i32* row = &weights[active[k] * OutputDimensions];
            for (IndexType o = 0; o < OutputDimensions; ++o)
                output[o] += row[o];
        }
    }

    bool read_parameters(std::istream& stream) {
        read_little_endian<i32>(stream, biases, OutputDimensions);
        read_little_endian<i32>(stream, weights, InputDimensions * OutputDimensions);
        return !stream.fail();
    }

    bool write_parameters(std::ostream& stream) const {
        write_little_endian<i32>(stream, biases, OutputDimensions);
        write_little_endian<i32>(stream, weights, InputDimensions * OutputDimensions);
        return !stream.fail();
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value());
        return h;
    }

   private:
    alignas(CacheLineSize) i32 biases[OutputDimensions];
    alignas(CacheLineSize) i32 weights[InputDimensions * OutputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_MATERIAL_INPUT_H_INCLUDED
