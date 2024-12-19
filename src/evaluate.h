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

#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include <string>

#include "position.h"
#include "smallnet.h"
#include "nnue/network.h"
#include "types.h"

namespace Stockfish {

namespace Eval {

// The default net name MUST follow the format nn-[SHA256 first 12 digits].nnue
// for the build process (profile-build and fishtest) to work. Do not change the
// name of the macro or the location where this macro is defined, as it is used
// in the Makefile/Fishtest.
#define EvalFileDefaultNameExtraBig "nn-84e2983ee6a6.nnue"
#define EvalFileDefaultNameBig "nn-1c0000000000.nnue"
#define EvalFileDefaultNameSmall "nn-37f18f62d772.nnue"

namespace NNUE {
struct Networks;
struct AccumulatorCaches;
}

std::string trace(Position& pos, const Eval::NNUE::Networks& networks);

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
template<NodeType nodeType>
Value evaluate(const Eval::NNUE::Networks&    networks,
               const Position&                pos,
               Eval::NNUE::AccumulatorCaches& caches,
               int                            optimism) {

    assert(!pos.checkers());

    constexpr bool PvNode   = nodeType != NonPV;
    bool           smallNet = !PvNode && use_smallnet(pos);
    auto [psqt, positional] = smallNet ? networks.small.evaluate(pos, &caches.small)
                            : PvNode   ? networks.extraBig.evaluate(pos, &caches.extraBig)
                                       : networks.big.evaluate(pos, &caches.big);

    Value nnue = (125 * psqt + 131 * positional) / 128;

    // Re-evaluate the position when higher eval accuracy is worth the time spent
    if (smallNet && (std::abs(nnue) < 236))
    {
        std::tie(psqt, positional) = networks.big.evaluate(pos, &caches.big);
        nnue                       = (125 * psqt + 131 * positional) / 128;
        smallNet                   = false;
    }

    // Blend optimism and eval with nnue complexity
    int nnueComplexity = std::abs(psqt - positional);
    optimism += optimism * nnueComplexity / 468;
    nnue -= nnue * nnueComplexity / (smallNet ? 20233 : 17879);

    int material = (smallNet ? 553 : 532) * pos.count<PAWN>() + pos.non_pawn_material();
    int v        = (nnue * (77777 + material) + optimism * (7777 + material)) / 77777;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 212;

    // Guarantee evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}
}  // namespace Eval

}  // namespace Stockfish

#endif  // #ifndef EVALUATE_H_INCLUDED
