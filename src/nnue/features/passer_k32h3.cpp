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


#include "passer_k32h3.h"

#include <cassert>
#include <cstdint>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "k32q2.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

// Row offset of the perspective's bucket, and the orientation of the squares
struct Frame {
    IndexType base;
    u8        orientation;
};

sf_always_inline Frame frame(Color perspective, Square ksq, int psqBucket) {
    const u8  flip = 56 * perspective;
    const int kb   = int(K32Q2::KingBuckets[int(ksq) ^ flip] / (2 * 11 * SQUARE_NB));
    assert(kb >= 0 && kb < 32);
    // Engine buckets: 2 = opponent queen, 1 = some rook or queen, 0 = none (see K32Q2::bucket())
    const int bucket = psqBucket == 0 ? 64 + kb : kb * 2 + (psqBucket == 2);
    return {IndexType(bucket * 96), u8(u8(K32Q2::OrientTBL[ksq]) ^ flip)};
}

template<bool Prefetch>
sf_always_inline void append_passers(Color                   perspective,
                                     const Frame&            f,
                                     Color                   c,
                                     Bitboard                b,
                                     PasserK32H3::IndexList& list,
                                     const ThreatWeightType* prefetchBase   = nullptr,
                                     IndexType               prefetchStride = 0) {
    const IndexType base = f.base + (c != perspective) * 48 - 8;
    while (b)
    {
        const IndexType index = base + (u8(pop_lsb(b)) ^ f.orientation);
        if constexpr (Prefetch)
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));
        list.push_back(index);
    }
}

}  // namespace

void PasserK32H3::append_active_indices(Color           perspective,
                                        const Position& pos,
                                        int             psqBucket,
                                        IndexList&      active) {
    const Frame    f     = frame(perspective, pos.square<KING>(perspective), psqBucket);
    const Bitboard white = pos.pieces(WHITE, PAWN), black = pos.pieces(BLACK, PAWN);
    for (Color c : {WHITE, BLACK})
        append_passers<false>(perspective, f, c, passed(c, white, black), active);
}

void PasserK32H3::append_changes(Color                   perspective,
                                 Square                  ksq,
                                 int                     psqBucket,
                                 const DirtyPawnPairs&   pawns,
                                 IndexList&              removed,
                                 IndexList&              added,
                                 const ThreatWeightType* prefetchBase,
                                 IndexType               prefetchStride) {
    const Frame f = frame(perspective, ksq, psqBucket);
    for (Color c : {WHITE, BLACK})
    {
        const Bitboard before = passed(c, pawns.before[WHITE], pawns.before[BLACK]);
        const Bitboard after  = passed(c, pawns.after[WHITE], pawns.after[BLACK]);
        if (before == after)
            continue;
        if (prefetchBase)
        {
            append_passers<true>(perspective, f, c, before & ~after, removed, prefetchBase,
                                 prefetchStride);
            append_passers<true>(perspective, f, c, after & ~before, added, prefetchBase,
                                 prefetchStride);
        }
        else
        {
            append_passers<false>(perspective, f, c, before & ~after, removed);
            append_passers<false>(perspective, f, c, after & ~before, added);
        }
    }
}

void PasserK32H3::append_king_move(Color                 perspective,
                                   Square                oldKsq,
                                   Square                newKsq,
                                   int                   psqBucket,
                                   const DirtyPawnPairs& pawns,
                                   IndexList&            removed,
                                   IndexList&            added) {
    const Frame fOld = frame(perspective, oldKsq, psqBucket);
    const Frame fNew = frame(perspective, newKsq, psqBucket);
    for (Color c : {WHITE, BLACK})
    {
        append_passers<false>(perspective, fOld, c,
                              passed(c, pawns.before[WHITE], pawns.before[BLACK]), removed);
        append_passers<false>(perspective, fNew, c,
                              passed(c, pawns.after[WHITE], pawns.after[BLACK]), added);
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
