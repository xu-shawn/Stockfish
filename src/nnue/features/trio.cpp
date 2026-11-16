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

#include <cassert>
#include <iterator>

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

// Incremental updates record rows as perspective-free descriptors: kind << 24 | fields, with AddBit for added rows
enum Kind : u32 {
    MOB_COVER,
    BLOCKADE,
    TRIANGLE
};
constexpr u32 AddBit = 1u << 31;
constexpr u32 desc(Kind k, u32 v) { return (u32(k) << 24) | v; }

Bitboard pawn_attacks(Color c, Bitboard b) {
    return c == WHITE ? pawn_attacks_bb<WHITE>(b) : pawn_attacks_bb<BLACK>(b);
}
Bitboard forward(Color c, Bitboard b) { return c == WHITE ? shift(b, NORTH) : shift(b, SOUTH); }

struct Tables {
    // Squares of enemy pawns covering a knight of that colour on the square, and the reverse
    Bitboard coverZone[COLOR_NB][SQUARE_NB];
    Bitboard coverRev[COLOR_NB][SQUARE_NB];
    // The four squares ahead of a bishop of that colour (one ahead, two ahead, two ahead on the
    // lower / higher file), and the bishops of that colour whose squares hold the square
    Bitboard triangle[COLOR_NB][SQUARE_NB][4];
    Bitboard triangleRev[COLOR_NB][SQUARE_NB];

    Tables() {
        auto sq = [](int f, int r) {
            return f < 0 || f > 7 || r < 0 || r > 7 ? Bitboard(0) : square_bb(Square(r * 8 + f));
        };
        for (Color c : {WHITE, BLACK})
        {
            const int dr = c == WHITE ? 1 : -1;
            for (Square s = SQ_A1; s <= SQ_H8; ++s)
                coverRev[c][s] = triangleRev[c][s] = 0;
            for (Square s = SQ_A1; s <= SQ_H8; ++s)
            {
                const int      f = file_of(s), r = rank_of(s);
                const Bitboard att = Attacks::attacks_bb(KNIGHT, s);
                coverZone[c][s]    = pawn_attacks(c, att) & ~pawn_attacks(c, square_bb(s)) & ~att;
                for (Bitboard b = coverZone[c][s]; b;)
                    coverRev[c][pop_lsb(b)] |= square_bb(s);
                triangle[c][s][0] = sq(f, r + dr);
                triangle[c][s][1] = sq(f, r + 2 * dr);
                triangle[c][s][2] = sq(f - 1, r + 2 * dr);
                triangle[c][s][3] = sq(f + 1, r + 2 * dr);
                for (int i = 0; i < 4; ++i)
                    if (triangle[c][s][i])
                        triangleRev[c][lsb(triangle[c][s][i])] |= square_bb(s);
            }
        }
    }
};
const Tables T;

// Pattern of enemy pawns e on the squares ahead of a bishop of colour c on s (bit 4: lower file)
u32 triangle_pattern(Color c, Square s, Bitboard e) {
    const Bitboard* t = T.triangle[c][s];
    return u32(bool(e & t[0])) | (u32(bool(e & t[1])) << 1) | (u32(bool(e & t[2])) << 2)
         | (u32(bool(e & t[3])) << 3);
}

// Index of a descriptor for one perspective
IndexType map(u32 x, Color perspective, unsigned orient) {
    const u32 v = x & 0xFFFFFF, p = perspective;
    switch (x >> 24)
    {
    case MOB_COVER :  // knight colour, knight square, pawn square
        return (((v >> 12) ^ p) * 64 + (((v >> 6) & 63) ^ orient)) * 48 + ((v & 63) ^ orient)
             - SQ_A2;
    case BLOCKADE :  // pawn colour, pawn square
        return Trio::MobCoverDimensions + ((v >> 6) ^ p) * 48 + ((v & 63) ^ orient) - SQ_A2;
    default : {  // bishop colour, bishop square, pattern with files mirrored when the orientation mirrors them
        u32 pat = v & 15;
        if ((orient & 7) == 7)
            pat = (pat & 3) | ((pat & 4) << 1) | ((pat & 8) >> 1);
        return Trio::MobCoverDimensions + Trio::BlockadeDimensions
             + (((v >> 10) ^ p) * 64 + (((v >> 4) & 63) ^ orient)) * 16 + pat;
    }
    }
}

}  // namespace

// Rows that exist with the piece on s are added by a put and removed by a removal; a bishop pattern
// without a pawn that is put or removed changes the other way round.
void Trio::piece_event(const Position& pos, Piece pc, bool put, Square s, DirtyTrio& out) {
    const Color    c    = color_of(pc);
    const Bitboard sb   = square_bb(s);
    auto           emit = [&](bool withPiece, u32 d) {
        assert(out.count < int(std::size(out.rows)));
        out.rows[out.count++] = d | (withPiece == put ? AddBit : 0);
    };

    switch (type_of(pc))
    {
    case KNIGHT :
        if (forward(~c, pos.pieces(~c, PAWN)) & sb)
            emit(true, desc(BLOCKADE, (u32(~c) << 6) | u32(s - (c == WHITE ? SOUTH : NORTH))));
        for (Bitboard b = pos.pieces(~c, PAWN) & T.coverZone[c][s]; b;)
            emit(true, desc(MOB_COVER, (u32(c) << 12) | (u32(s) << 6) | u32(pop_lsb(b))));
        break;
    case BISHOP :
        if (const u32 pat = triangle_pattern(c, s, pos.pieces(~c, PAWN)))
            emit(true, desc(TRIANGLE, (u32(c) << 10) | (u32(s) << 4) | pat));
        break;
    case PAWN : {
        if (pos.pieces(~c, KNIGHT) & forward(c, sb))
            emit(true, desc(BLOCKADE, (u32(c) << 6) | u32(s)));
        for (Bitboard b = pos.pieces(~c, KNIGHT) & T.coverRev[~c][s]; b;)
            emit(true, desc(MOB_COVER, (u32(~c) << 12) | (u32(pop_lsb(b)) << 6) | u32(s)));
        const Bitboard pawns = pos.pieces(c, PAWN);
        for (Bitboard b = pos.pieces(~c, BISHOP) & T.triangleRev[~c][s]; b;)
        {
            const Square bs  = pop_lsb(b);
            const u32    key = (u32(~c) << 10) | (u32(bs) << 4);
            emit(true, desc(TRIANGLE, key | triangle_pattern(~c, bs, pawns | sb)));
            if (const u32 without = triangle_pattern(~c, bs, pawns & ~sb))
                emit(false, desc(TRIANGLE, key | without));
        }
        break;
    }
    default :
        break;
    }
}

void Trio::cancel_events(DirtyTrio& out) {
    int n = out.count;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if ((out.rows[i] ^ out.rows[j]) == AddBit)
            {
                out.rows[j] = out.rows[--n];
                out.rows[i] = out.rows[--n];
                --i;
                break;
            }
    out.count = n;
}

void Trio::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    TrioBitboards bb;
    for (Color c : {WHITE, BLACK})
        for (PieceType pt : {PAWN, KNIGHT, BISHOP})
            bb[c][pt - PAWN] = pos.pieces(c, pt);
    append_indices(perspective, pos.square<KING>(perspective), bb, active);
}

void Trio::append_changed_indices(
  Color perspective, Square ksq, const DiffType& diff, IndexList& removed, IndexList& added) {
    const unsigned orient = OrientTBL[ksq] ^ (56 * perspective);
    for (int i = 0; i < diff.count; ++i)
    {
        const u32 x = diff.rows[i];
        ((x & AddBit) ? added : removed).push_back(map(x & ~AddBit, perspective, orient));
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
