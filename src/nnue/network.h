/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "../memory.h"
#include "../position.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_feature_transformer.h"
#include "nnue_misc.h"

namespace Stockfish::Eval::NNUE {

enum class EmbeddedNNUEType {
    EXTRA_BIG,
    BIG,
    SMALL,
};

using NetworkOutput = std::tuple<Value, Value>;

template<typename Arch, typename Transformer>
class Network {
    static constexpr IndexType FTDimensions = Arch::TransformedFeatureDimensions;

   public:
    Network(EvalFile file, EmbeddedNNUEType type) :
        evalFile(file),
        embeddedType(type) {}

    Network(const Network& other);
    Network(Network&& other) = default;

    Network& operator=(const Network& other);
    Network& operator=(Network&& other) = default;

    void load(const std::string& rootDirectory, std::string evalfilePath);
    bool save(const std::optional<std::string>& filename) const;

    NetworkOutput evaluate(const Position&                         pos,
                           AccumulatorCaches::Cache<FTDimensions>* cache) const;


    void hint_common_access(const Position&                         pos,
                            AccumulatorCaches::Cache<FTDimensions>* cache) const;

    void verify(std::string evalfilePath, const std::function<void(std::string_view)>&) const;
    NnueEvalTrace trace_evaluate(const Position&                         pos,
                                 AccumulatorCaches::Cache<FTDimensions>* cache) const;

   private:
    void load_user_net(const std::string&, const std::string&);
    void load_internal();

    void initialize();

    bool                       save(std::ostream&, const std::string&, const std::string&) const;
    std::optional<std::string> load(std::istream&);

    bool read_header(std::istream&, std::uint32_t*, std::string*) const;
    bool write_header(std::ostream&, std::uint32_t, const std::string&) const;

    bool read_parameters(std::istream&, std::string&) const;
    bool write_parameters(std::ostream&, const std::string&) const;

    // Input feature converter
    LargePagePtr<Transformer> featureTransformer;

    // Evaluation function
    AlignedPtr<Arch[]> network;

    EvalFile         evalFile;
    EmbeddedNNUEType embeddedType;

    // Hash value of evaluation function structure
    static constexpr std::uint32_t hash = Transformer::get_hash_value() ^ Arch::get_hash_value();

    template<IndexType Size>
    friend struct AccumulatorCaches::Cache;
};

// Definitions of the network types
using SmallFeatureTransformer =
  FeatureTransformer<TransformedFeatureDimensionsSmall, &StateInfo::accumulatorSmall>;
using SmallNetworkArchitecture =
  NetworkArchitecture<TransformedFeatureDimensionsSmall, L2Small, L3Small>;

using BigFeatureTransformer =
  FeatureTransformer<TransformedFeatureDimensionsBig, &StateInfo::accumulatorBig>;
using BigNetworkArchitecture = NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>;

using ExtraBigFeatureTransformer =
  FeatureTransformer<TransformedFeatureDimensionsExtraBig, &StateInfo::accumulatorExtraBig>;
using ExtraBigNetworkArchitecture =
  NetworkArchitecture<TransformedFeatureDimensionsExtraBig, L2ExtraBig, L3ExtraBig>;

using NetworkExtraBig = Network<ExtraBigNetworkArchitecture, ExtraBigFeatureTransformer>;
using NetworkBig      = Network<BigNetworkArchitecture, BigFeatureTransformer>;
using NetworkSmall    = Network<SmallNetworkArchitecture, SmallFeatureTransformer>;


struct Networks {
    Networks(NetworkExtraBig&& nxB, NetworkBig&& nB, NetworkSmall&& nS) :
        extraBig(std::move(nxB)),
        big(std::move(nB)),
        small(std::move(nS)) {}

    NetworkExtraBig extraBig;
    NetworkBig      big;
    NetworkSmall    small;
};

template<NodeType nodeType>
void hint_common_parent_position(const Position&    pos,
                                 const Networks&    networks,
                                 AccumulatorCaches& caches) {
    if constexpr (nodeType == PV)
        networks.extraBig.hint_common_access(pos, &caches.extraBig);
    if (use_smallnet(pos))
        networks.small.hint_common_access(pos, &caches.small);
    else
        networks.big.hint_common_access(pos, &caches.big);
}

}  // namespace Stockfish

#endif
