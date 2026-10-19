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

//Definition of input features Stop_Threats of NNUE evaluation function

#include "stop_threats.h"

#include <array>
#include <cassert>
#include <cstdint>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

// First feature of each (pawn color relative to the perspective, oriented stop square,
// attacker kind), numbered like the trainer: by pawn color, pawn square and kind, each
// kind taking one feature per square from which it reaches the stop square. Pawns of
// color 0 advance towards the oriented 8th rank.
struct StopOffsets {
    std::array<std::array<std::array<u16, 10>, SQUARE_NB>, COLOR_NB> first;
    int                                                              total;
};

constexpr StopOffsets init_stop_offsets() {
    StopOffsets t{};
    int         offset = 0;

    for (int c = 0; c < COLOR_NB; ++c)
        for (Square pawn = SQ_A2; pawn <= SQ_H7; ++pawn)
        {
            const Square stop = c == 0 ? pawn + NORTH : pawn + SOUTH;
            for (int kind = 0; kind < 10; ++kind)
            {
                t.first[c][stop][kind] = u16(offset);
                offset += constexpr_popcount(Attacks::PseudoAttacks[KNIGHT + kind % 5][stop]);
            }
        }

    t.total = offset;
    return t;
}

constexpr StopOffsets stopOffsets = init_stop_offsets();
static_assert(stopOffsets.total == StopThreats::Dimensions);

}  // namespace

// Index of a feature for a given king position: attacker on from, attacking the empty
// stop square of a pawn
sf_always_inline IndexType StopThreats::make_index(
  Color perspective, Piece pawn, Square stop, Piece attacker, Square from, Square ksq) {
    assert(type_of(pawn) == PAWN && type_of(attacker) >= KNIGHT);

    const u8        orientation   = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);
    const Square    stop_oriented = Square(u8(stop) ^ orientation);
    const Square    from_oriented = Square(u8(from) ^ orientation);
    const PieceType pt            = type_of(attacker);
    const int       kind          = pt - KNIGHT + 5 * (color_of(attacker) != color_of(pawn));

    return stopOffsets.first[color_of(pawn) ^ perspective][stop_oriented][kind]
         + popcount(Attacks::PseudoAttacks[pt][stop_oriented] & (square_bb(from_oriented) - 1));
}

void StopThreats::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square   ksq      = pos.square<KING>(perspective);
    const Bitboard occupied = pos.pieces();
    const Bitboard nonPawns = occupied ^ pos.pieces(PAWN);
    const Bitboard passed   = passed_pawns_bb(pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN));

    for (Color c : {WHITE, BLACK})
    {
        const Piece pawn = make_piece(c, PAWN);
        for (Bitboard stops = shift(pos.pieces(c) & passed, pawn_push(c)) & ~occupied; stops;)
        {
            const Square stop = pop_lsb(stops);
            for (Bitboard b = pos.attackers_to(stop, occupied) & nonPawns; b;)
            {
                const Square from = pop_lsb(b);
                active.push_back(
                  make_index(perspective, pawn, stop, pos.piece_on(from), from, ksq));
            }
        }
    }
}

// Changed features are recorded as attacker -> pawn threats with the stop square as the
// threatened square (see Position::update_piece_threats and Position::do_move)

void StopThreats::append_changed_indices(Color                   perspective,
                                         Square                  ksq,
                                         const DiffType&         diff,
                                         IndexList&              removed,
                                         IndexList&              added,
                                         const ThreatWeightType* prefetchBase,
                                         IndexType               prefetchStride) {

    for (const auto& dirty : diff.stops)
    {
        const IndexType index = make_index(perspective, dirty.threatened_pc(),
                                           dirty.threatened_sq(), dirty.pc(), dirty.pc_sq(), ksq);

        if (prefetchBase)
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));

        (dirty.add() ? added : removed).push_back(index);
    }
}

void StopThreats::append_changed_indices_both(Square                  white_ksq,
                                              Square                  black_ksq,
                                              const DiffType&         diff,
                                              IndexList&              white_removed,
                                              IndexList&              white_added,
                                              IndexList&              black_removed,
                                              IndexList&              black_added,
                                              const ThreatWeightType* prefetchBase,
                                              IndexType               prefetchStride) {

    for (const auto& dirty : diff.stops)
    {
        const Piece  pawn     = dirty.threatened_pc();
        const Square stop     = dirty.threatened_sq();
        const Piece  attacker = dirty.pc();
        const Square from     = dirty.pc_sq();

        const IndexType white_index = make_index(WHITE, pawn, stop, attacker, from, white_ksq);
        const IndexType black_index = make_index(BLACK, pawn, stop, attacker, from, black_ksq);

        if (prefetchBase)
        {
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(white_index) * prefetchStride));
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(black_index) * prefetchStride));
        }

        (dirty.add() ? white_added : white_removed).push_back(white_index);
        (dirty.add() ? black_added : black_removed).push_back(black_index);
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
