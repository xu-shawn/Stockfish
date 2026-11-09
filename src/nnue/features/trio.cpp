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


#include "trio.h"

#include <algorithm>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../position.h"

namespace Stockfish::Eval::NNUE::Features {

namespace {

// Pawn, knight and bishop bitboards of both colours
using TrioBitboards = Bitboard[COLOR_NB][3];

// Per-perspective caps of active features, as in the trainer
constexpr int MaxMobCover = 64, MaxBlockade = 8, MaxTriangle = 10;

void append_indices(Color perspective, Square ksq, const TrioBitboards& bb, Trio::IndexList& out) {
    const unsigned orient    = OrientTBL[ksq] ^ (56 * perspective);
    int            nMobCover = 0, nBlockade = 0, nTriangle = 0;

    for (Color c : {WHITE, BLACK})
    {
        const Color    them    = ~c;
        const unsigned rel     = c ^ perspective;
        const Bitboard pawns   = bb[c][0];
        const Bitboard enemyP  = bb[them][0];
        const Bitboard knights = bb[c][1];

        for (Bitboard b = knights; b;)
        {
            const Square   s   = pop_lsb(b);
            const Bitboard att = Attacks::attacks_bb(KNIGHT, s);
            // Enemy pawns attacking a square of att, but not the knight, and not attacked by it
            Bitboard cover = c == WHITE
                             ? pawn_attacks_bb<WHITE>(att) & ~pawn_attacks_bb<WHITE>(square_bb(s))
                             : pawn_attacks_bb<BLACK>(att) & ~pawn_attacks_bb<BLACK>(square_bb(s));
            cover &= enemyP & ~att;
            const IndexType row = (rel * 64 + (unsigned(s) ^ orient)) * 48;
            for (; cover && nMobCover < MaxMobCover; ++nMobCover)
                out.push_back(row + (unsigned(pop_lsb(cover)) ^ orient) - SQ_A2);
        }

        const Bitboard blocked =
          pawns & (c == WHITE ? shift(bb[them][1], SOUTH) : shift(bb[them][1], NORTH));
        for (Bitboard b = blocked; b && nBlockade < MaxBlockade; ++nBlockade)
            out.push_back(Trio::MobCoverDimensions + rel * 48 + (unsigned(pop_lsb(b)) ^ orient)
                          - SQ_A2);

        const int dr = c == WHITE ? 1 : -1;
        for (Bitboard b = bb[c][2]; b;)
        {
            const Square   s = pop_lsb(b);
            const int      f = file_of(s), r = rank_of(s);
            const unsigned os      = unsigned(s) ^ orient;
            auto           pawn_at = [&](int ff, int rr) {
                return ff >= 0 && ff < 8 && rr >= 0 && rr < 8
                    && (enemyP & square_bb(Square(rr * 8 + ff)));
            };
            int pat = 0;
            if (pawn_at(f, r + dr))
                pat |= 1;
            if (pawn_at(f, r + 2 * dr))
                pat |= 2;
            for (int df : {-1, 1})
                if (pawn_at(f + df, r + 2 * dr))
                    pat |= ((((r + 2 * dr) * 8 + f + df) ^ orient) % 8 > os % 8) ? 8 : 4;
            if (pat && nTriangle++ < MaxTriangle)
                out.push_back(Trio::MobCoverDimensions + Trio::BlockadeDimensions
                              + (rel * 64 + os) * 16 + pat);
        }
    }
}

}  // namespace

void Trio::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    TrioBitboards bb;
    for (Color c : {WHITE, BLACK})
        for (PieceType pt : {PAWN, KNIGHT, BISHOP})
            bb[c][pt - PAWN] = pos.pieces(c, pt);
    append_indices(perspective, pos.square<KING>(perspective), bb, active);
}

void Trio::append_changed_indices(
  Color perspective, Square ksq, const DiffType& diff, IndexList& removed, IndexList& added) {
    if (std::equal(&diff.before[0][0], &diff.before[0][0] + 6, &diff.after[0][0]))
        return;

    IndexList before, after;
    append_indices(perspective, ksq, diff.before, before);
    append_indices(perspective, ksq, diff.after, after);

    for (IndexType i : before)
        if (std::find(after.begin(), after.end(), i) == after.end())
            removed.push_back(i);
    for (IndexType i : after)
        if (std::find(before.begin(), before.end(), i) == before.end())
            added.push_back(i);
}

}  // namespace Stockfish::Eval::NNUE::Features
