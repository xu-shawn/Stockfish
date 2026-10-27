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


#include "ram_ram.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

// The ram given by its white pawn w, numbered by the oriented square of its lower pawn
sf_always_inline IndexType ram_id(Color perspective, u8 orientation, Square w) {
    const Square lower = perspective == WHITE ? w : w + NORTH;
    assert((u8(lower) ^ orientation) >= SQ_A2 && (u8(lower) ^ orientation) <= SQ_H6);
    return (u8(lower) ^ orientation) - 8;
}

sf_always_inline IndexType pair_index(IndexType a, IndexType b) {
    const IndexType hi = std::max(a, b);
    const IndexType lo = std::min(a, b);
    return RamRam::IndexBase + hi * (hi - 1) / 2 + lo;
}

// Pairs every ram of first with every ram of others, and with the rams of first after it,
// so that each pair within first appears once
template<bool Prefetch>
sf_always_inline void append_pairs(Color                   perspective,
                                   u8                      orientation,
                                   Bitboard                first,
                                   Bitboard                others,
                                   RamRam::IndexList&      list,
                                   const ThreatWeightType* prefetchBase   = nullptr,
                                   IndexType               prefetchStride = 0) {
    while (first)
    {
        const Square    w  = pop_lsb(first);
        const IndexType id = ram_id(perspective, orientation, w);

        for (Bitboard b = others | first; b;)
        {
            const IndexType index = pair_index(id, ram_id(perspective, orientation, pop_lsb(b)));
            if constexpr (Prefetch)
                prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
                  reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));
            list.push_back(index);
        }
    }
}

// The squares on the file of the ram given by its white pawn w and on the adjacent files
sf_always_inline Bitboard near_files(Square w) {
    const Bitboard f = file_bb(w);
    return f | shift(f, EAST) | shift(f, WEST);
}

// The Ram_Pawn_Near features of the ram given by its white pawn w with the white pawns of white
// and the black pawns of black, which must not include the ram's own pawns
template<bool Prefetch>
sf_always_inline void append_ram_pawns(Color                   perspective,
                                       u8                      orientation,
                                       Square                  w,
                                       Bitboard                white,
                                       Bitboard                black,
                                       RamPawnNear::IndexList& list,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0) {
    const IndexType base = ram_id(perspective, orientation, w) * 96;

    for (Color c : {WHITE, BLACK})
        for (Bitboard b = c == WHITE ? white : black; b;)
        {
            const IndexType index =
              base + (c != perspective) * 48 + (u8(pop_lsb(b)) ^ orientation) - 8;
            if constexpr (Prefetch)
                prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
                  reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));
            list.push_back(index);
        }
}

template<bool Prefetch>
void append_ram_pawn_changes(Color                   perspective,
                             u8                      orientation,
                             const DirtyPawnPairs&   pawns,
                             RamPawnNear::IndexList& removed,
                             RamPawnNear::IndexList& added,
                             const ThreatWeightType* prefetchBase,
                             IndexType               prefetchStride) {
    const Bitboard before = RamRam::rams_of(pawns.before[WHITE], pawns.before[BLACK]);
    const Bitboard after  = RamRam::rams_of(pawns.after[WHITE], pawns.after[BLACK]);

    // A broken ram loses all its features, a formed one gains all of them
    for (Bitboard b = before & ~after; b;)
    {
        const Square   w      = pop_lsb(b);
        const Bitboard others = near_files(w) & ~(square_bb(w) | square_bb(w + NORTH));
        append_ram_pawns<Prefetch>(perspective, orientation, w, pawns.before[WHITE] & others,
                                   pawns.before[BLACK] & others, removed, prefetchBase,
                                   prefetchStride);
    }
    for (Bitboard b = after & ~before; b;)
    {
        const Square   w      = pop_lsb(b);
        const Bitboard others = near_files(w) & ~(square_bb(w) | square_bb(w + NORTH));
        append_ram_pawns<Prefetch>(perspective, orientation, w, pawns.after[WHITE] & others,
                                   pawns.after[BLACK] & others, added, prefetchBase,
                                   prefetchStride);
    }

    // The pawns of a kept ram stay, so it loses the pawns that left its files and gains
    // those that arrived
    const Bitboard leftWhite    = pawns.before[WHITE] & ~pawns.after[WHITE];
    const Bitboard leftBlack    = pawns.before[BLACK] & ~pawns.after[BLACK];
    const Bitboard arrivedWhite = pawns.after[WHITE] & ~pawns.before[WHITE];
    const Bitboard arrivedBlack = pawns.after[BLACK] & ~pawns.before[BLACK];
    const Bitboard moved        = leftWhite | leftBlack | arrivedWhite | arrivedBlack;

    for (Bitboard b = before & after; b;)
    {
        const Square   w    = pop_lsb(b);
        const Bitboard near = near_files(w);
        if (!(moved & near))
            continue;
        append_ram_pawns<Prefetch>(perspective, orientation, w, leftWhite & near, leftBlack & near,
                                   removed, prefetchBase, prefetchStride);
        append_ram_pawns<Prefetch>(perspective, orientation, w, arrivedWhite & near,
                                   arrivedBlack & near, added, prefetchBase, prefetchStride);
    }
}

}  // namespace

void RamRam::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square ksq         = pos.square<KING>(perspective);
    const u8     orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);

    append_pairs<false>(perspective, orientation,
                        rams_of(pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN)), 0, active);
}

void RamRam::append_ram_changes(Color                   perspective,
                                Square                  ksq,
                                const DirtyPawnPairs&   pawns,
                                IndexList&              removed,
                                IndexList&              added,
                                const ThreatWeightType* prefetchBase,
                                IndexType               prefetchStride) {
    const u8       orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);
    const Bitboard before      = rams_of(pawns.before[WHITE], pawns.before[BLACK]);
    const Bitboard after       = rams_of(pawns.after[WHITE], pawns.after[BLACK]);
    const Bitboard kept        = before & after;

    // The pairs that involve a broken ram, and those that involve a formed one
    if (prefetchBase)
    {
        append_pairs<true>(perspective, orientation, before & ~after, kept, removed, prefetchBase,
                           prefetchStride);
        append_pairs<true>(perspective, orientation, after & ~before, kept, added, prefetchBase,
                           prefetchStride);
    }
    else
    {
        append_pairs<false>(perspective, orientation, before & ~after, kept, removed);
        append_pairs<false>(perspective, orientation, after & ~before, kept, added);
    }
}

void RamPawnNear::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square   ksq         = pos.square<KING>(perspective);
    const u8       orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);
    const Bitboard white       = pos.pieces(WHITE, PAWN);
    const Bitboard black       = pos.pieces(BLACK, PAWN);

    for (Bitboard b = RamRam::rams_of(white, black); b;)
    {
        const Square   w      = pop_lsb(b);
        const Bitboard others = near_files(w) & ~(square_bb(w) | square_bb(w + NORTH));
        append_ram_pawns<false>(perspective, orientation, w, white & others, black & others,
                                active);
    }
}

void RamPawnNear::append_changes(Color                   perspective,
                                 Square                  ksq,
                                 const DirtyPawnPairs&   pawns,
                                 IndexList&              removed,
                                 IndexList&              added,
                                 const ThreatWeightType* prefetchBase,
                                 IndexType               prefetchStride) {
    const u8 orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);

    if (prefetchBase)
        append_ram_pawn_changes<true>(perspective, orientation, pawns, removed, added, prefetchBase,
                                      prefetchStride);
    else
        append_ram_pawn_changes<false>(perspective, orientation, pawns, removed, added, nullptr, 0);
}

}  // namespace Stockfish::Eval::NNUE::Features
