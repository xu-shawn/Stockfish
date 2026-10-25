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

#include "attacked_state.h"

#include <cstdint>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

// raw: square | piece << 6 | state << 10
sf_always_inline IndexType make_index(Color perspective, u8 orientation, u32 raw) {
    const Square s  = Square(raw & 63);
    const Piece  pc = Piece((raw >> 6) & 15);
    return AttackedState::IndexBase
         + (((color_of(pc) != perspective) * 5 + (type_of(pc) - PAWN)) * 64 + (u8(s) ^ orientation)) * 6
         + (raw >> 10);
}

}  // namespace

void AttackedState::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    if (pos.count<ALL_PIECES>() - pos.count<PAWN>() - 2 > AttackedMaxPieces)
        return;

    const u8       orientation = u8(FullThreats::OrientTBL[pos.square<KING>(perspective)]) ^ u8(56 * perspective);
    const Bitboard occupied    = pos.pieces();
    for (Bitboard b = occupied ^ pos.pieces(KING); b;)
    {
        const Square s    = pop_lsb(b);
        const u8     code = pos.attacked_code(s, occupied);
        if (code)
            active.push_back(make_index(perspective, orientation,
                                        u32(s) | u32(code & 15) << 6 | u32((code >> 4) - 1) << 10));
    }
}

void AttackedState::append_attacked_changes(Color                   perspective,
                                            Square                  ksq,
                                            const DirtyAttacked&    a,
                                            IndexList&              removed,
                                            IndexList&              added,
                                            const ThreatWeightType* prefetchBase,
                                            IndexType               prefetchStride) {
    const u8 orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);
    for (u16 raw : a.removed)
        removed.push_back(make_index(perspective, orientation, raw));
    for (u16 raw : a.added)
    {
        const IndexType index = make_index(perspective, orientation, raw);
        if (prefetchBase)
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));
        added.push_back(index);
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
