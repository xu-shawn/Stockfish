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

#include "xray_threats.h"

#include <array>
#include <cstdint>
#include <cstdlib>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

// Pair index of (attacker square, target square) among the lines of a bishop, rook or queen
// at Chebyshev distance >= 2, in the oriented frame; numbering as in the trainer's loader
struct XRayLines {
    std::array<std::array<std::array<u16, SQUARE_NB>, SQUARE_NB>, 3> pair{};
    std::array<IndexType, 6>                                          offset{};

    constexpr XRayLines() {
        IndexType count[3] = {0, 0, 0};
        for (int ti = 0; ti < 3; ++ti)
            for (int a = 0; a < SQUARE_NB; ++a)
                for (int b = 0; b < SQUARE_NB; ++b)
                {
                    const int  df   = (b & 7) - (a & 7), dr = (b >> 3) - (a >> 3);
                    const int  adf  = df < 0 ? -df : df, adr = dr < 0 ? -dr : dr;
                    const bool diag = df != 0 && adf == adr;
                    const bool orth = (df == 0) != (dr == 0);
                    const bool ok   = ti == 0 ? diag : ti == 1 ? orth : (diag || orth);
                    pair[ti][a][b]  = ok && (adf > adr ? adf : adr) >= 2 ? u16(count[ti]++) : u16(0xFFFF);
                }
        IndexType off = 0;
        for (int kind = 0; kind < 6; ++kind)
        {
            offset[kind] = off;
            off += count[kind % 3] * 24;
        }
    }
};

constexpr XRayLines Lines;
static_assert(Lines.offset[5] + XRayThreats::LinePairs[2] * 24 == XRayThreats::Dimensions);

sf_always_inline IndexType make_index(Color perspective, u8 orientation, u32 raw) {
    const Square from = Square(raw & 63), to = Square((raw >> 6) & 63);
    const Piece  sp = Piece((raw >> 12) & 15), tp = Piece((raw >> 16) & 15);
    const int    ti   = type_of(sp) - BISHOP;
    const int    kind = (color_of(sp) != perspective) * 3 + ti;
    const u16    pi   = Lines.pair[ti][u8(from) ^ orientation][u8(to) ^ orientation];
    assert(pi != 0xFFFF);
    return XRayThreats::IndexBase + Lines.offset[kind] + IndexType(pi) * 24
         + ((color_of(tp) != perspective) * 6 + (type_of(tp) - PAWN)) * 2 + ((raw >> 20) & 1);
}

}  // namespace

void XRayThreats::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    if (pos.count<ALL_PIECES>() - pos.count<PAWN>() - 2 > XRayMaxPieces)
        return;

    const u8 orientation = u8(FullThreats::OrientTBL[pos.square<KING>(perspective)]) ^ u8(56 * perspective);
    u32      list[DirtyXRays::MaxSize];
    const int n = pos.xray_list(list);
    for (int i = 0; i < n; ++i)
        active.push_back(make_index(perspective, orientation, list[i]));
}

void XRayThreats::append_xray_changes(Color                   perspective,
                                      Square                  ksq,
                                      const DirtyXRays&       x,
                                      IndexList&              removed,
                                      IndexList&              added,
                                      const ThreatWeightType* prefetchBase,
                                      IndexType               prefetchStride) {
    const u8 orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);
    for (u32 raw : x.removed)
        removed.push_back(make_index(perspective, orientation, raw));
    for (u32 raw : x.added)
    {
        const IndexType index = make_index(perspective, orientation, raw);
        if (prefetchBase)
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));
        added.push_back(index);
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
