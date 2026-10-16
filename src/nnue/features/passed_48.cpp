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


#include "passed_48.h"

#include <cstdint>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

// Pawns of color c with no enemy pawn ahead on the same or an adjacent file
Bitboard passed_pawns(Color c, Bitboard white, Bitboard black) {
    const Bitboard enemy = c == WHITE ? black : white;
    const Bitboard span  = enemy | shift(enemy, WEST) | shift(enemy, EAST);

    Bitboard ahead;
    if (c == WHITE)
    {
        ahead = span >> 8;
        ahead |= ahead >> 8;
        ahead |= ahead >> 16;
        ahead |= ahead >> 32;
    }
    else
    {
        ahead = span << 8;
        ahead |= ahead << 8;
        ahead |= ahead << 16;
        ahead |= ahead << 32;
    }

    return (c == WHITE ? white : black) & ~ahead;
}

sf_always_inline IndexType make_index(Square s, u8 orientation) {
    return Passed48::IndexBase + (u8(s) ^ orientation) - 8;
}

}  // namespace

void Passed48::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const u8 orientation =
      u8(FullThreats::OrientTBL[pos.square<KING>(perspective)]) ^ u8(56 * perspective);

    for (Bitboard b = passed_pawns(perspective, pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN));
         b;)
        active.push_back(make_index(pop_lsb(b), orientation));
}

void Passed48::append_changed_indices(Color                   perspective,
                                      Square                  ksq,
                                      const Dirties&          diff,
                                      IndexList&              removed,
                                      IndexList&              added,
                                      const ThreatWeightType* prefetchBase,
                                      IndexType               prefetchStride) {
    const auto& pawns = diff.dirtyPawnPairs;

    if (pawns.before[WHITE] == pawns.after[WHITE] && pawns.before[BLACK] == pawns.after[BLACK])
        return;

    const Bitboard before = passed_pawns(perspective, pawns.before[WHITE], pawns.before[BLACK]);
    const Bitboard after  = passed_pawns(perspective, pawns.after[WHITE], pawns.after[BLACK]);

    if (before == after)
        return;

    const u8 orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);

    auto push = [&](IndexList& out, Bitboard b) {
        while (b)
        {
            const IndexType index = make_index(pop_lsb(b), orientation);
            if (prefetchBase)
                prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
                  reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));
            out.push_back(index);
        }
    };

    push(removed, before & ~after);
    push(added, after & ~before);
}

}  // namespace Stockfish::Eval::NNUE::Features
