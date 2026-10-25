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

// Definition of input features XRay_Threats_N4 of NNUE evaluation function: slider x-rays.
// Every bishop, rook or queen whose line, past exactly one piece (the blocker), reaches a
// second piece (the target), while there are at most XRayMaxPieces knights, bishops, rooks
// and queens on the board. Covers pins, skewers, discovered attacks and batteries, which the
// threat features (direct attacks only) do not.

#ifndef NNUE_FEATURES_XRAY_THREATS_INCLUDED
#define NNUE_FEATURES_XRAY_THREATS_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"
#include "pp_3wide.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class XRayThreats {
   public:
    static constexpr u32 HashValue = 0x58a40004u;

    // Attacker kind (bishop, rook, queen of the perspective, then of the opponent), the
    // (attacker, target) square pair among the lines of that piece type at distance at
    // least 2, target kind (pawn ... king of the perspective, then of the opponent) and
    // whether the blocker belongs to the other side than the attacker
    static constexpr IndexType LinePairs[3] = {364, 672, 1036};
    static constexpr IndexType Dimensions   = 2 * (LinePairs[0] + LinePairs[1] + LinePairs[2]) * 24;

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
        const DirtyXRays& x = diff.dirtyXRays;
        if (x.removed.size() || x.added.size())
            append_xray_changes(perspective, ksq, x, removed, added, prefetchBase,
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
        const DirtyXRays& x = diff.dirtyXRays;
        if (x.removed.size() || x.added.size())
        {
            append_xray_changes(WHITE, white_ksq, x, white_removed, white_added, prefetchBase,
                                prefetchStride);
            append_xray_changes(BLACK, black_ksq, x, black_removed, black_added, prefetchBase,
                                prefetchStride);
        }
    }

   private:
    static void append_xray_changes(Color                   perspective,
                                    Square                  ksq,
                                    const DirtyXRays&       x,
                                    IndexList&              removed,
                                    IndexList&              added,
                                    const ThreatWeightType* prefetchBase,
                                    IndexType               prefetchStride);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_XRAY_THREATS_INCLUDED
