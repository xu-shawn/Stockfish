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
#include "../../misc.h"
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
//
// At load time the weights are folded into two tables: one row per joint signature (with
// the bias, the N, B, R, Q counts and their differences, which the signature determines
// whenever no side has more than two knights, bishops or rooks or more than one queen) and
// one row per pawn counts and bishop pairs. An evaluation then adds two rows instead of up
// to 17; positions beyond those counts (promotions) take the general path. Integer sums are
// exact, so both paths give the same result.
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

    // Identifies the material input of a position: material counts (the material key), side
    // to move and bishop pairs, which piece counts alone do not determine
    static Key cache_key(const Position& pos) {
        constexpr Bitboard Light    = 0x55AA55AA55AA55AAULL;
        const Bitboard     bishops0 = pos.pieces(pos.side_to_move(), BISHOP);
        const Bitboard     bishops1 = pos.pieces(~pos.side_to_move(), BISHOP);
        const Key          pairs    = (bool(bishops0 & Light) & bool(bishops0 & ~Light))
                       | (bool(bishops1 & Light) & bool(bishops1 & ~Light)) << 1;
        return pos.material_key() ^ (pos.side_to_move() * 0x9E3779B97F4A7C15ULL)
             ^ (pairs * 0xC2B2AE3D27D4EB4FULL);
    }

    // Writes biases plus the weights of the active features to output
    void propagate(const Position& pos, OutputType* RESTRICT output) const {
        constexpr Bitboard Light = 0x55AA55AA55AA55AAULL;

        const Color us = pos.side_to_move(), them = ~us;
        const int   n0 = pos.count<KNIGHT>(us), n1 = pos.count<KNIGHT>(them);
        const int   b0 = pos.count<BISHOP>(us), b1 = pos.count<BISHOP>(them);
        const int   r0 = pos.count<ROOK>(us), r1 = pos.count<ROOK>(them);
        const int   q0 = pos.count<QUEEN>(us), q1 = pos.count<QUEEN>(them);

        if (std::max({n0, n1, b0, b1, r0, r1}) > 2 || std::max(q0, q1) > 1)
        {
            propagate_general(pos, output);
            return;
        }

        const Bitboard bishops0 = pos.pieces(us, BISHOP), bishops1 = pos.pieces(them, BISHOP);
        const int      pair0    = bool(bishops0 & Light) & bool(bishops0 & ~Light);
        const int      pair1    = bool(bishops1 & Light) & bool(bishops1 & ~Light);
        const int      pawns =
          ((pos.count<PAWN>(us) * 9 + pos.count<PAWN>(them)) * 2 + pair0) * 2 + pair1;
        const int signature = (q0 * 27 + r0 * 9 + b0 * 3 + n0) * 54 + q1 * 27 + r1 * 9 + b1 * 3 + n1;

        const i32* sigRow  = &signatureTable[signature * OutputDimensions];
        const i32* pawnRow = &pawnTable[pawns * OutputDimensions];
        for (IndexType o = 0; o < OutputDimensions; ++o)
            output[o] = sigRow[o] + pawnRow[o];
    }

    // Any material: biases plus the rows of all active features
    void propagate_general(const Position& pos, OutputType* output) const {
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
        build_tables();
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
    // Feature offsets, in the order of active_features()
    static constexpr int OwnCounts = 0, OppCounts = 20, MinorDiff = 40, RookDiff = 47,
                         QueenDiff = 52, PawnDiff = 55, BishopPair = 64, Signature = 66;

    void add_row(i32* dst, int feature) const {
        for (IndexType o = 0; o < OutputDimensions; ++o)
            dst[o] += weights[feature * OutputDimensions + o];
    }

    void build_tables() {
        for (int sig0 = 0; sig0 < 54; ++sig0)
            for (int sig1 = 0; sig1 < 54; ++sig1)
            {
                // signature = Q * 27 + R * 9 + B * 3 + N, counts within the caps
                const int q0 = sig0 / 27, r0 = sig0 / 9 % 3, b0 = sig0 / 3 % 3, n0 = sig0 % 3;
                const int q1 = sig1 / 27, r1 = sig1 / 9 % 3, b1 = sig1 / 3 % 3, n1 = sig1 % 3;
                i32*      row = &signatureTable[(sig0 * 54 + sig1) * OutputDimensions];
                for (IndexType o = 0; o < OutputDimensions; ++o)
                    row[o] = biases[o];
                add_row(row, Signature + sig0 * 54 + sig1);
                add_row(row, OwnCounts + 9 + n0);
                add_row(row, OwnCounts + 12 + b0);
                add_row(row, OwnCounts + 15 + r0);
                add_row(row, OwnCounts + 18 + q0);
                add_row(row, OppCounts + 9 + n1);
                add_row(row, OppCounts + 12 + b1);
                add_row(row, OppCounts + 15 + r1);
                add_row(row, OppCounts + 18 + q1);
                add_row(row, MinorDiff + std::clamp(n0 + b0 - n1 - b1, -3, 3) + 3);
                add_row(row, RookDiff + std::clamp(r0 - r1, -2, 2) + 2);
                add_row(row, QueenDiff + std::clamp(q0 - q1, -1, 1) + 1);
            }

        for (int p0 = 0; p0 <= 8; ++p0)
            for (int p1 = 0; p1 <= 8; ++p1)
                for (int pair0 = 0; pair0 < 2; ++pair0)
                    for (int pair1 = 0; pair1 < 2; ++pair1)
                    {
                        i32* row = &pawnTable[(((p0 * 9 + p1) * 2 + pair0) * 2 + pair1)
                                              * OutputDimensions];
                        for (IndexType o = 0; o < OutputDimensions; ++o)
                            row[o] = 0;
                        add_row(row, OwnCounts + p0);
                        add_row(row, OppCounts + p1);
                        add_row(row, PawnDiff + std::clamp(p0 - p1, -4, 4) + 4);
                        if (pair0)
                            add_row(row, BishopPair);
                        if (pair1)
                            add_row(row, BishopPair + 1);
                    }
    }

    alignas(CacheLineSize) i32 biases[OutputDimensions];
    alignas(CacheLineSize) i32 weights[InputDimensions * OutputDimensions];

    // Derived at load time (see above)
    alignas(CacheLineSize) i32 signatureTable[54 * 54 * OutputDimensions];
    alignas(CacheLineSize) i32 pawnTable[9 * 9 * 2 * 2 * OutputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_MATERIAL_INPUT_H_INCLUDED
