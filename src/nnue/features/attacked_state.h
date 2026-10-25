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

// Definition of input features Attacked_State of NNUE evaluation function: every non-king piece
// attacked by at least one enemy piece, with its attacker and defender counts (direct attacks,
// kings and pawns included), while there are at most AttackedMaxPieces knights, bishops, rooks
// and queens on the board. Index: ((piece color relative to the perspective * 5 + piece type)
// * 64 + oriented square) * 6 + (min(attackers, 2) - 1) * 3 + min(defenders, 2).

#ifndef NNUE_FEATURES_ATTACKED_STATE_INCLUDED
#define NNUE_FEATURES_ATTACKED_STATE_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"
#include "pp_3wide.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class AttackedState {
   public:
    static constexpr u32 HashValue =
      AttackedMaxPieces >= 30 ? 0xa77ac456u : 0xa77ac400u + u32(AttackedMaxPieces);

    static constexpr IndexType Dimensions = 10 * 64 * 6;

    // Concatenated to threats and pawn pairs; see nnue_feature_transformer.h
    static constexpr IndexType IndexBase = FullThreats::Dimensions + PP_3Wide::Dimensions;
    using IndexList                      = FullThreats::IndexList;

    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    static void append_changed_indices(Color                   perspective,
                                       Square                  ksq,
                                       const Dirties&          diff,
                                       IndexList&              removed,
                                       IndexList&              added,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0) {
        const DirtyAttacked& a = diff.dirtyAttacked;
        if (a.removed.size() || a.added.size())
            append_attacked_changes(perspective, ksq, a, removed, added, prefetchBase,
                                    prefetchStride);
    }

    static void append_changed_indices_both(Square                  white_ksq,
                                            Square                  black_ksq,
                                            const Dirties&          diff,
                                            IndexList&              white_removed,
                                            IndexList&              white_added,
                                            IndexList&              black_removed,
                                            IndexList&              black_added,
                                            const ThreatWeightType* prefetchBase,
                                            IndexType               prefetchStride) {
        const DirtyAttacked& a = diff.dirtyAttacked;
        if (a.removed.size() || a.added.size())
        {
            append_attacked_changes(WHITE, white_ksq, a, white_removed, white_added, prefetchBase,
                                    prefetchStride);
            append_attacked_changes(BLACK, black_ksq, a, black_removed, black_added, prefetchBase,
                                    prefetchStride);
        }
    }

   private:
    static void append_attacked_changes(Color                   perspective,
                                        Square                  ksq,
                                        const DirtyAttacked&    a,
                                        IndexList&              removed,
                                        IndexList&              added,
                                        const ThreatWeightType* prefetchBase,
                                        IndexType               prefetchStride);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_ATTACKED_STATE_INCLUDED
