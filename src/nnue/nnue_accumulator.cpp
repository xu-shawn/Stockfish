#include "nnue_accumulator.h"

namespace Stockfish::Eval::NNUE {

void AccumulatorState::reset(const DirtyPiece& dp) noexcept {
    dirtyPiece = dp;
    accumulatorBig.computed.fill(false);
    accumulatorSmall.computed.fill(false);
}

}
