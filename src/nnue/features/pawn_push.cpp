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


#include "pawn_push.h"

#include <cassert>
#include <cstdint>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"

namespace Stockfish::Eval::NNUE::Features {

sf_always_inline IndexType
PawnPush::make_index(Color perspective, Piece pawn, Square from, Piece blocker, Square ksq) {
    assert(type_of(pawn) == PAWN && type_of(blocker) != PAWN);

    const u8        orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);
    const IndexType pawnColor   = color_of(pawn) ^ perspective;
    const IndexType kind = (color_of(blocker) ^ perspective) * 5 + (type_of(blocker) - KNIGHT);

    return IndexBase + (pawnColor * 10 + kind) * 48 + (u8(from) ^ orientation) - 8;
}

void PawnPush::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square   ksq      = pos.square<KING>(perspective);
    const Bitboard blockers = pos.pieces() ^ pos.pieces(PAWN);

    for (Color c : {WHITE, BLACK})
    {
        const Piece pawn = make_piece(c, PAWN);
        for (Bitboard b = shift(pos.pieces(c, PAWN), pawn_push(c)) & blockers; b;)
        {
            const Square to = pop_lsb(b);
            active.push_back(
              make_index(perspective, pawn, to - pawn_push(c), pos.piece_on(to), ksq));
        }
    }
}

void PawnPush::append_push_changes(Color                   perspective,
                                   Square                  ksq,
                                   const Dirties&          diff,
                                   IndexList&              removed,
                                   IndexList&              added,
                                   const ThreatWeightType* prefetchBase,
                                   IndexType               prefetchStride) {

    for (const auto& dirty : diff.dirtyThreats.pushes)
    {
        const IndexType index =
          make_index(perspective, dirty.pc(), dirty.pc_sq(), dirty.threatened_pc(), ksq);

        if (prefetchBase)
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));

        (dirty.add() ? added : removed).push_back(index);
    }
}

void PawnPush::append_push_changes_both(Square                  white_ksq,
                                        Square                  black_ksq,
                                        const Dirties&          diff,
                                        IndexList&              white_removed,
                                        IndexList&              white_added,
                                        IndexList&              black_removed,
                                        IndexList&              black_added,
                                        const ThreatWeightType* prefetchBase,
                                        IndexType               prefetchStride) {

    for (const auto& dirty : diff.dirtyThreats.pushes)
    {
        const IndexType white_index =
          make_index(WHITE, dirty.pc(), dirty.pc_sq(), dirty.threatened_pc(), white_ksq);
        const IndexType black_index =
          make_index(BLACK, dirty.pc(), dirty.pc_sq(), dirty.threatened_pc(), black_ksq);

        prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
          reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(white_index) * prefetchStride));
        prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
          reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(black_index) * prefetchStride));

        (dirty.add() ? white_added : white_removed).push_back(white_index);
        (dirty.add() ? black_added : black_removed).push_back(black_index);
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
