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
    static constexpr u32 HashValue = 0x3a9e1c03u;

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

// Ram_Pawn_Near: every ram with every other pawn (either colour) on the ram's file or an adjacent
// file. The features depend only on the pawns. Their weights follow those of Ram_Ram in the int8
// threat weight table, but the threat, pawn-pair and Ram_Ram indices already fill most of the u16
// range, so they have an index list of their own, relative to their first row (WeightRow).
class RamPawnNear {
   public:
    static constexpr u32 HashValue = 0x3a9e1c11u;

    // Ram id * 96 + (pawn colour != perspective) * 48 + oriented square of the pawn - 8
    static constexpr IndexType Dimensions = RamRam::RamIds * COLOR_NB * 48;

    // First weight row in the threat weight table; see nnue_feature_transformer.h
    static constexpr IndexType WeightRow = RamRam::IndexBase + RamRam::Dimensions;

    // At most 8 rams, each with at most the 14 other pawns
    static constexpr IndexType MaxActiveDimensions = 112;
    using IndexList                                = ValueList<u16, MaxActiveDimensions>;

    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    // Pawn moves away from the rams leave the features unchanged
    static void append_changed_indices(Color                   perspective,
                                       Square                  ksq,
                                       const DirtyPawnPairs&   pawns,
                                       IndexList&              removed,
                                       IndexList&              added,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0) {
        if (may_change(pawns))
            append_changes(perspective, ksq, pawns, removed, added, prefetchBase, prefetchStride);
    }

    static void append_changed_indices_both(Square                  white_ksq,
                                            Square                  black_ksq,
                                            const DirtyPawnPairs&   pawns,
                                            IndexList&              white_removed,
                                            IndexList&              white_added,
                                            IndexList&              black_removed,
                                            IndexList&              black_added,
                                            const ThreatWeightType* prefetchBase,
                                            IndexType               prefetchStride) {
        if (may_change(pawns))
        {
            append_changes(WHITE, white_ksq, pawns, white_removed, white_added, prefetchBase,
                           prefetchStride);
            append_changes(BLACK, black_ksq, pawns, black_removed, black_added, prefetchBase,
                           prefetchStride);
        }
    }

   private:
    // The files of the pieces of b, as an 8-bit mask
    static constexpr u8 file_mask(Bitboard b) {
        b |= b >> 32;
        b |= b >> 16;
        b |= b >> 8;
        return u8(b);
    }

    // The features change only if a ram forms or breaks, or a pawn leaves or reaches a square on
    // the file of a ram or an adjacent file
    static bool may_change(const DirtyPawnPairs& pawns) {
        const Bitboard before = RamRam::rams_of(pawns.before[WHITE], pawns.before[BLACK]);
        const Bitboard after  = RamRam::rams_of(pawns.after[WHITE], pawns.after[BLACK]);
        const Bitboard moved =
          (pawns.before[WHITE] ^ pawns.after[WHITE]) | (pawns.before[BLACK] ^ pawns.after[BLACK]);
        const u8 ramFiles = file_mask(before);
        return before != after
            || (file_mask(moved) & u8(ramFiles | (ramFiles << 1) | (ramFiles >> 1)));
    }

    static void append_changes(Color                   perspective,
                               Square                  ksq,
                               const DirtyPawnPairs&   pawns,
                               IndexList&              removed,
                               IndexList&              added,
                               const ThreatWeightType* prefetchBase,
                               IndexType               prefetchStride);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_RAM_RAM_INCLUDED
