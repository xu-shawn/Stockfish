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


// Knight_MobCover, Knight_Blockade and Bishop_Triangle input features, concatenated in that order
//   Knight_MobCover: opponent pawns attacking one of a knight's attack squares, except pawns attacking
//   the knight and pawns it attacks (6144 rows)
//   Knight_Blockade: pawns whose stop square holds an enemy knight (96 rows)
//   Bishop_Triangle: per bishop, which of the four squares ahead of it hold enemy pawns (2048 rows)

#ifndef NNUE_FEATURES_TRIO_INCLUDED
#define NNUE_FEATURES_TRIO_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class Trio {
   public:
    static constexpr u32 MobCoverHashValue = 0x4b4d4332u;
    static constexpr u32 BlockadeHashValue = 0x4b424b32u;
    static constexpr u32 TriangleHashValue = 0x42545231u;

    static constexpr IndexType MobCoverDimensions = 2 * 64 * 48;
    static constexpr IndexType BlockadeDimensions = 2 * 48;
    static constexpr IndexType TriangleDimensions = 2 * 64 * 16;
    static constexpr IndexType Dimensions =
      MobCoverDimensions + BlockadeDimensions + TriangleDimensions;

    // Piece types whose puts and removals can change rows
    static constexpr u32 EventTypes = (1u << PAWN) | (1u << KNIGHT) | (1u << BISHOP);

    static constexpr IndexType MaxActiveDimensions = 128;
    using IndexList                                = ValueList<u16, MaxActiveDimensions>;
    using DiffType                                 = DirtyTrio;

    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    static void append_changed_indices(
      Color perspective, Square ksq, const DiffType& diff, IndexList& removed, IndexList& added);

    // Records the rows involving piece pc on square s (pc is on the board), called by Position for every
    // piece put and removal during do_move
    static void piece_event(const Position& pos, Piece pc, bool put, Square s, DirtyTrio& out);

    // Drops the rows a move both removed and added
    static void cancel_events(DirtyTrio& out);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_TRIO_INCLUDED
