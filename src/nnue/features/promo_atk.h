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


// Definition of input features Promo_Atk of NNUE evaluation function: for each
// pawn on its 7th rank, the attackers of its promotion square, counted per
// attacker kind relative to the pawn (own pawn, knight, bishop, rook, queen,
// king, enemy knight, bishop, rook, queen, king) with two thermometer levels,
// one or more attackers and two or more.

#ifndef NNUE_FEATURES_PROMO_ATK_INCLUDED
#define NNUE_FEATURES_PROMO_ATK_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"
#include "pp_3wide.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class PromoAtk {
   public:
    static constexpr u32 HashValue = 0x7d29a4c3u;

    // Pawn color, promotion file, and the 22 bits of DirtyPromo::State
    static constexpr IndexType Dimensions = COLOR_NB * FILE_NB * DirtyPromo::Bits;

    // Concatenated to threats and pawn pairs; see nnue_feature_transformer.h
    static constexpr IndexType IndexBase = FullThreats::Dimensions + PP_3Wide::Dimensions;
    using IndexList                      = ValueList<u16, 256>;

    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    // Nothing to do unless a pawn stands on its 7th rank before or after the move
    static void append_changed_indices(Color                   perspective,
                                       Square                  ksq,
                                       const Dirties&          diff,
                                       IndexList&              removed,
                                       IndexList&              added,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0) {
        const auto& pawns = diff.dirtyPawnPairs;
        if (has_seventh_rank_pawn(pawns.before[WHITE] | pawns.after[WHITE],
                                  pawns.before[BLACK] | pawns.after[BLACK]))
            append_promo_changes(perspective, ksq, diff, removed, added, prefetchBase,
                                 prefetchStride);
    }

   private:
    static void append_promo_changes(Color                   perspective,
                                     Square                  ksq,
                                     const Dirties&          diff,
                                     IndexList&              removed,
                                     IndexList&              added,
                                     const ThreatWeightType* prefetchBase,
                                     IndexType               prefetchStride);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_PROMO_ATK_INCLUDED
