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


#include "promo_atk.h"

#include <cstdint>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

template<typename Push>
sf_always_inline void
append_bits(Color perspective, Square ksq, Color c, File f, u32 bits, const Push& push) {
    const IndexType file = f ^ (u8(FullThreats::OrientTBL[ksq]) & 7);
    const IndexType base =
      PromoAtk::IndexBase + ((c ^ perspective) * FILE_NB + file) * DirtyPromo::Bits;

    while (bits)
    {
        push(base + IndexType(lsb(Bitboard(bits))));
        bits &= bits - 1;
    }
}

}  // namespace

void PromoAtk::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    if (!has_seventh_rank_pawn(pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN)))
        return;

    DirtyPromo::State state;
    pos.promo_attacks(state);

    const Square ksq = pos.square<KING>(perspective);
    for (Color c : {WHITE, BLACK})
        for (File f = FILE_A; f <= FILE_H; ++f)
            append_bits(perspective, ksq, c, f, state[c][f],
                        [&](IndexType index) { active.push_back(index); });
}

void PromoAtk::append_promo_changes(Color                   perspective,
                                    Square                  ksq,
                                    const Dirties&          diff,
                                    IndexList&              removed,
                                    IndexList&              added,
                                    const ThreatWeightType* prefetchBase,
                                    IndexType               prefetchStride) {
    const auto& promo     = diff.dirtyPromo;
    const auto& pawns     = diff.dirtyPawnPairs;
    const bool  hasBefore = has_seventh_rank_pawn(pawns.before[WHITE], pawns.before[BLACK]);
    const bool  hasAfter  = has_seventh_rank_pawn(pawns.after[WHITE], pawns.after[BLACK]);

    auto pusher = [&](IndexList& out) {
        return [&, out_ptr = &out](IndexType index) {
            if (prefetchBase)
                prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
                  reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));
            out_ptr->push_back(index);
        };
    };

    for (Color c : {WHITE, BLACK})
        for (File f = FILE_A; f <= FILE_H; ++f)
        {
            const u32 before = hasBefore ? (*promo.before)[c][f] : 0;
            const u32 after  = hasAfter ? (*promo.after)[c][f] : 0;
            if (before == after)
                continue;

            append_bits(perspective, ksq, c, f, before & ~after, pusher(removed));
            append_bits(perspective, ksq, c, f, after & ~before, pusher(added));
        }
}

}  // namespace Stockfish::Eval::NNUE::Features
