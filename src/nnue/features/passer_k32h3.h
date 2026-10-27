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


// Definition of input features Passer_K32H3 of NNUE evaluation function (speed prototype): every
// passed pawn (no enemy pawn ahead on its own or an adjacent file), indexed by the perspective's
// K32H3 bucket (king bucket and heavy / opponent-queen state, as the piece-square features), the
// pawn's colour and its square, oriented like the piece-square features. The features depend on
// the pawns and the bucket; bucket changes refresh the accumulator anyway.

#ifndef NNUE_FEATURES_PASSER_K32H3_INCLUDED
#define NNUE_FEATURES_PASSER_K32H3_INCLUDED

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "ram_ram.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class PasserK32H3 {
   public:
    static constexpr u32 HashValue = 0x9a55e2e3u;

    // K32H3 bucket * 96 + (pawn colour != perspective) * 48 + oriented square - 8, where the K32H3
    // bucket is king bucket * 2 + opponent queen with some rook or queen on the board, else
    // 64 + king bucket
    static constexpr IndexType Dimensions = 96 * 96;

    // First weight row in the threat weight table; see nnue_feature_transformer.h
    static constexpr IndexType WeightRow = RamRam::IndexBase + RamRam::Dimensions;

    static constexpr IndexType MaxActiveDimensions = 16;
    using IndexList                                = ValueList<u16, MaxActiveDimensions>;

    // Passed pawns of white and black
    static Bitboard passed(Color c, Bitboard white, Bitboard black) {
        Bitboard front = c == WHITE ? black >> 8 : white << 8;
        front |= c == WHITE ? front >> 8 : front << 8;
        front |= c == WHITE ? front >> 16 : front << 16;
        front |= c == WHITE ? front >> 32 : front << 32;
        front |= shift(front, EAST) | shift(front, WEST);
        return (c == WHITE ? white : black) & ~front;
    }

    static void
    append_active_indices(Color perspective, const Position& pos, int psqBucket, IndexList& active);

    static void append_changed_indices(Color                   perspective,
                                       Square                  ksq,
                                       int                     psqBucket,
                                       const DirtyPawnPairs&   pawns,
                                       IndexList&              removed,
                                       IndexList&              added,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0) {
        if (pawns.before[WHITE] != pawns.after[WHITE] || pawns.before[BLACK] != pawns.after[BLACK])
            append_changes(perspective, ksq, psqBucket, pawns, removed, added, prefetchBase,
                           prefetchStride);
    }

    static void append_changed_indices_both(Square                  white_ksq,
                                            Square                  black_ksq,
                                            int                     whiteBucket,
                                            int                     blackBucket,
                                            const DirtyPawnPairs&   pawns,
                                            IndexList&              white_removed,
                                            IndexList&              white_added,
                                            IndexList&              black_removed,
                                            IndexList&              black_added,
                                            const ThreatWeightType* prefetchBase,
                                            IndexType               prefetchStride) {
        if (pawns.before[WHITE] != pawns.after[WHITE] || pawns.before[BLACK] != pawns.after[BLACK])
        {
            append_changes(WHITE, white_ksq, whiteBucket, pawns, white_removed, white_added,
                           prefetchBase, prefetchStride);
            append_changes(BLACK, black_ksq, blackBucket, pawns, black_removed, black_added,
                           prefetchBase, prefetchStride);
        }
    }

    // A king move within its half: every passer moves to the new king bucket
    static void append_king_move(Color                 perspective,
                                 Square                oldKsq,
                                 Square                newKsq,
                                 int                   psqBucket,
                                 const DirtyPawnPairs& pawns,
                                 IndexList&            removed,
                                 IndexList&            added);

   private:
    static void append_changes(Color                   perspective,
                               Square                  ksq,
                               int                     psqBucket,
                               const DirtyPawnPairs&   pawns,
                               IndexList&              removed,
                               IndexList&              added,
                               const ThreatWeightType* prefetchBase,
                               IndexType               prefetchStride);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_PASSER_K32H3_INCLUDED
