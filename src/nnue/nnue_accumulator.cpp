#include "nnue_accumulator.h"

#include "../position.h"

namespace Stockfish::Eval::NNUE {

namespace {

template<IndexType Size>
int refresh_cost(const typename AccumulatorCaches::Cache<Size>::Entry& entry, const Position& pos) {
    int cost = 0;
    for (Color c : {WHITE, BLACK})
    {
        for (PieceType pt = PAWN; pt <= KING; ++pt)
        {
            const Bitboard oldBB    = entry.byColorBB[c] & entry.byTypeBB[pt];
            const Bitboard newBB    = pos.pieces(c, pt);
            Bitboard       toRemove = oldBB & ~newBB;
            Bitboard       toAdd    = newBB & ~oldBB;

            cost += popcount(toRemove) + popcount(toAdd);
        }
    }
    return cost;
}

}

// To initialize a refresh entry, we set all its bitboards empty,
// so we put the biases in the accumulation, without any weights on top
template<IndexType Size>
void AccumulatorCaches::Cache<Size>::Entry::clear(const BiasType* biases) {
    std::memcpy(accumulation, biases, sizeof(accumulation));
    std::memset((uint8_t*) this + offsetof(Entry, psqtAccumulation), 0,
                sizeof(Entry) - offsetof(Entry, psqtAccumulation));
}

template<IndexType Size>
typename AccumulatorCaches::Cache<Size>::EntryPair AccumulatorCaches::Cache<Size>::get(
  const Square ksq, const Color perspective, const Position& pos) {
    auto&       avaliable  = entries[ksq][perspective];
    int         least_cost = 128;
    int         most_cost  = 0;
    std::size_t best_entry_idx;
    std::size_t worst_entry_idx;

    for (std::size_t i = 0; i < Duplication; i++)
    {
        const auto cost = refresh_cost<Size>(avaliable[i], pos);

        if (cost < least_cost)
        {
            least_cost     = cost;
            best_entry_idx = i;
        }

        if (cost >= most_cost)
        {
            most_cost       = cost;
            worst_entry_idx = i;
        }
    }

    return {avaliable[best_entry_idx], avaliable[worst_entry_idx]};
}

// Explicit template instantiations
template struct AccumulatorCaches::Cache<TransformedFeatureDimensionsBig>;
template struct AccumulatorCaches::Cache<TransformedFeatureDimensionsSmall>;

}
