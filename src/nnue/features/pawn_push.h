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


// Definition of input features Pawn_Push of NNUE evaluation function: a pawn
// whose push square holds a non-pawn piece of either color. Pawns blocked by
// pawns are covered by PP_3Wide.

#ifndef NNUE_FEATURES_PAWN_PUSH_INCLUDED
#define NNUE_FEATURES_PAWN_PUSH_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"
#include "pp_3wide.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class PawnPush {
   public:
    static constexpr u32 HashValue = 0x5c3a91e7u;

    // Pawn color, blocker color, blocker type (knight to king) and pawn square
    static constexpr IndexType Dimensions = COLOR_NB * COLOR_NB * 5 * 48;

    // Concatenated to threats and pawn pairs; see nnue_feature_transformer.h
    static constexpr IndexType IndexBase = FullThreats::Dimensions + PP_3Wide::Dimensions;
    using IndexList                      = ValueList<u16, 256>;

    static IndexType
    make_index(Color perspective, Piece pawn, Square from, Piece blocker, Square ksq);

    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    static void append_changed_indices(Color                   perspective,
                                       Square                  ksq,
                                       const Dirties&          diff,
                                       IndexList&              removed,
                                       IndexList&              added,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_PAWN_PUSH_INCLUDED
