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

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

static int simple_eval(const Position& pos) {
    const Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c)) + pos.non_pawn_material(c)
         - pos.non_pawn_material(~c);
}

Value scale_evaluation(Value nnue, int optimism, const Position& pos);

namespace {

// Endgame eval map, fitted to the results of 43k fishtest games at 60+0.6 (1.55M quiet positions,
// nn-c9e480e92425). With few pieces left, small advantages are drawn more often, and large ones
// converted more often, than the same eval means elsewhere. Row: piece count (kings included,
// 16 and up share the last row). Entry k: the mapped value of |v| = 50 * k, linear in between.
constexpr int EndgameMapStep                  = 50;
constexpr int EndgameMapKnots                 = 13;
constexpr int EndgameMap[15][EndgameMapKnots] = {
  {0, 25, 51, 82, 116, 158, 205, 260, 322, 390, 463, 541, 623},    // 2 pieces
  {0, 25, 51, 82, 116, 158, 205, 260, 322, 390, 463, 541, 623},    // 3 pieces
  {0, 25, 51, 82, 116, 158, 205, 260, 322, 390, 463, 541, 623},    // 4 pieces
  {0, 26, 54, 85, 121, 164, 213, 269, 333, 402, 477, 556, 639},    // 5 pieces
  {0, 28, 58, 91, 130, 176, 228, 288, 354, 427, 505, 586, 672},    // 6 pieces
  {0, 30, 62, 98, 139, 188, 243, 306, 376, 452, 532, 617, 704},    // 7 pieces
  {0, 32, 65, 103, 147, 197, 254, 318, 389, 465, 546, 631, 718},   // 8 pieces
  {0, 33, 69, 108, 152, 202, 260, 323, 393, 468, 547, 629, 715},   // 9 pieces
  {0, 36, 74, 115, 160, 212, 269, 332, 400, 473, 550, 629, 712},   // 10 pieces
  {0, 39, 80, 124, 171, 224, 281, 343, 410, 481, 554, 631, 710},   // 11 pieces
  {0, 43, 87, 133, 182, 236, 293, 354, 419, 486, 557, 629, 704},   // 12 pieces
  {0, 46, 93, 142, 194, 247, 304, 364, 426, 490, 556, 624, 694},   // 13 pieces
  {0, 50, 100, 152, 205, 259, 315, 373, 433, 494, 556, 619, 684},  // 14 pieces
  {0, 52, 104, 156, 210, 265, 321, 378, 436, 496, 556, 617, 679},  // 15 pieces
  {0, 52, 104, 156, 210, 265, 321, 378, 436, 496, 556, 617, 679},  // 16 pieces
};

int material_scale(int v, int material) { return v + int(i64(material) * v / 90649); }

// Up to 16 pieces the map replaces the material scaling, from 17 to 20 pieces it blends into it,
// and from 21 pieces on only the material scaling applies. Beyond the last knot the map continues
// with the material scaling's slope.
int scale_by_material(int v, int material, int pieces) {
    const int scaled = material_scale(v, material);
    if (pieces > 20)
        return scaled;

    const auto& row  = EndgameMap[std::min(pieces, 16) - 2];
    const int   a    = std::abs(v);
    const int   last = EndgameMapStep * (EndgameMapKnots - 1);
    int         m;
    if (a <= last)
    {
        const int i = std::min(a / EndgameMapStep, EndgameMapKnots - 2);
        m           = row[i] + (row[i + 1] - row[i]) * (a - i * EndgameMapStep) / EndgameMapStep;
    }
    else
        m = row[EndgameMapKnots - 1] + std::abs(scaled) - material_scale(last, material);

    const int t = std::clamp(pieces - 16, 0, 5);
    const int r = (m * (5 - t) + std::abs(scaled) * t) / 5;
    return v < 0 ? -r : r;
}

}  // namespace

Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());
    Value nnue = network.evaluate(pos, accumulators, caches);
    return scale_evaluation(nnue, optimism, pos);
}

// Applies search-dependent scaling (optimism and rule50) to the raw NNUE eval
Value scale_evaluation(Value nnue, int optimism, const Position& pos) {
    Value se = simple_eval(pos);

    // Normalize the raw evaluations to [-1024, 1024] to measure their correlation.
    int se_norm   = (se * 1024) / (std::abs(se) + 1024);
    int nnue_norm = (nnue * 1024) / (std::abs(nnue) + 1024);
    // When NNUE and material agree (positive correlation), the position is straightforward;
    // otherwise (negative correlation) it involves complex compensation. In a representative
    // sample, correl averages -1 or so, i.e. it is well-centered in [-2048, 2048].
    int correl = (se_norm * nnue_norm) / 512;

    // When winning, we boost easy positions, and vice versa
    int v = nnue + (nnue + optimism) * correl / 65536;

    // Scale the combined evaluation by total material, and reshape it in endgames
    int material = 521 * pos.count<PAWN>() + pos.non_pawn_material();
    v            = scale_by_material(v, material, pos.count<ALL_PIECES>());

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 189;

    // Guarantee that the evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Network& network) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(network);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, network, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    Value nnue = network.evaluate(pos, *accumulators, *caches);
    Value s_v  = scale_evaluation(nnue, VALUE_ZERO, pos);  // requires stm perspective

    ss << "NNUE evaluation          " << nnue << " (side to move, internal units)\n";

    nnue = pos.side_to_move() == WHITE ? nnue : -nnue;
    s_v  = pos.side_to_move() == WHITE ? s_v : -s_v;

    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(nnue, pos) << " (white side)\n";
    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(s_v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
