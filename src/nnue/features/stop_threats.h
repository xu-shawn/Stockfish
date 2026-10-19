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

// Definition of input features PassStop_Threats_N8 of NNUE evaluation function: threats on
// the empty square in front of every passed pawn (its stop square), where a passed pawn
// has no enemy pawn ahead of it on its own or an adjacent file, while there are at most
// eight knights, bishops, rooks and queens on the board (PassStopMaxPieces). Every knight,
// bishop, rook, queen or king attacking the square is a feature of its own, indexed by pawn
// color, pawn square, attacker kind and attacker square like Stop_Threats, which has them
// for every pawn. Pawn attackers are covered by PP_3Wide, attacks on an occupied stop
// square by the threat features.

#ifndef NNUE_FEATURES_STOP_THREATS_INCLUDED
#define NNUE_FEATURES_STOP_THREATS_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"
#include "full_threats.h"
#include "pp_3wide.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

class StopThreats {
   public:
    static constexpr u32 HashValue = 0x7d29a4f9u;

    // Pawn color, pawn square, attacker kind (own knight, bishop, rook, queen, king, then
    // the enemy's) and the squares from which that kind reaches the stop square
    static constexpr IndexType Dimensions = 11256;

    // The weights follow the threats and pawn pairs in the int8 weight array. Indices in
    // the lists are relative to IndexBase, so that they fit in u16.
    static constexpr IndexType IndexBase = FullThreats::Dimensions + PP_3Wide::Dimensions;

    // A square has at most 18 non-pawn attackers (8 knights, a slider on each of its 8
    // rays and 2 kings) and there are at most 16 pawns, so at most 288 features are
    // active, and a move removes or adds at most 288 (see DirtyStopList).
    using IndexList = ValueList<u16, 288>;
    using DiffType  = DirtyThreats;

    static IndexType
    make_index(Color perspective, Piece pawn, Square stop, Piece attacker, Square from, Square ksq);

    // Get a list of indices for active features
    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    // Get a list of indices for recently changed features
    static void append_changed_indices(Color                   perspective,
                                       Square                  ksq,
                                       const DiffType&         diff,
                                       IndexList&              removed,
                                       IndexList&              added,
                                       const ThreatWeightType* prefetchBase   = nullptr,
                                       IndexType               prefetchStride = 0);

    static void append_changed_indices_both(Square                  white_ksq,
                                            Square                  black_ksq,
                                            const DiffType&         diff,
                                            IndexList&              white_removed,
                                            IndexList&              white_added,
                                            IndexList&              black_removed,
                                            IndexList&              black_added,
                                            const ThreatWeightType* prefetchBase   = nullptr,
                                            IndexType               prefetchStride = 0);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_STOP_THREATS_INCLUDED
