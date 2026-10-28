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


// Definition of input features Ram_Ram of NNUE evaluation function: every pair of rams
// on the board, where a ram is a pawn blocked head-on by an enemy pawn. The features
// depend only on the pawns and change only when a ram forms or breaks.

#ifndef NNUE_FEATURES_RAM_RAM_INCLUDED
#define NNUE_FEATURES_RAM_RAM_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"
#include "pp_3wide.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class RamRam {
   public:
    static constexpr u32 HashValue = 0x5e9fcac8u;

    // Both ram ids (the oriented square of the ram's lower pawn, ranks 2 to 6), as a
    // triangular index over the pair
    static constexpr IndexType RamIds     = 40;
    static constexpr IndexType Dimensions = RamIds * (RamIds - 1) / 2;

    // Concatenated to threats and pawn pairs; see nnue_feature_transformer.h
    static constexpr IndexType IndexBase = FullThreats::Dimensions + PP_3Wide::Dimensions;
    using IndexList                      = ValueList<u16, 256>;

    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    // Most moves leave the rams unchanged
    static void append_changed_indices(Color                   perspective,
                                       Square                  ksq,
                                       const Dirties&          diff,
                                       IndexList&              removed,
                                       IndexList&              added,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0) {
        const DirtyPawnPairs& pawns = diff.dirtyPawnPairs;
        if (rams_changed(pawns))
            append_ram_changes(perspective, ksq, pawns, removed, added, prefetchBase,
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
        const DirtyPawnPairs& pawns = diff.dirtyPawnPairs;
        if (rams_changed(pawns))
        {
            append_ram_changes(WHITE, white_ksq, pawns, white_removed, white_added, prefetchBase,
                               prefetchStride);
            append_ram_changes(BLACK, black_ksq, pawns, black_removed, black_added, prefetchBase,
                               prefetchStride);
        }
    }

    // White pawns with a black pawn directly in front
    static constexpr Bitboard rams_of(Bitboard white, Bitboard black) {
        return white & (black >> 8);
    }

   private:
    static bool rams_changed(const DirtyPawnPairs& pawns) {
        return rams_of(pawns.before[WHITE], pawns.before[BLACK])
            != rams_of(pawns.after[WHITE], pawns.after[BLACK]);
    }

    static void append_ram_changes(Color                   perspective,
                                   Square                  ksq,
                                   const DirtyPawnPairs&   pawns,
                                   IndexList&              removed,
                                   IndexList&              added,
                                   const ThreatWeightType* prefetchBase,
                                   IndexType               prefetchStride);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_RAM_RAM_INCLUDED
