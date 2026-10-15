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

#ifndef ENGINE_H_INCLUDED
#define ENGINE_H_INCLUDED

#include <functional>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "misc.h"
#include "history.h"
#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "numa.h"
#include "position.h"
#include "search.h"
#include "syzygy/tbprobe.h"  // for Stockfish::Depth
#include "thread.h"
#include "tt.h"
#include "types.h"
#include "ucioption.h"

namespace Stockfish {

constexpr int MaxHashMB = Is64Bit ? 33554432 : 2048;
extern int    MaxThreads;

class Engine {
   public:
    using InfoShort = Search::InfoShort;
    using InfoFull  = Search::InfoFull;
    using InfoIter  = Search::InfoIteration;

    Engine(std::optional<std::filesystem::path> path = std::nullopt);

    // Cannot be movable due to components holding backreferences to fields
    Engine(const Engine&)            = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(const Engine&) = delete;
    Engine& operator=(Engine&&)      = delete;

    ~Engine() { wait_for_search_finished(); }

    std::variant<u64, PositionSetError> perft(const std::string& fen, Depth depth, bool isChess960);

    // non blocking call to start searching
    void go(Search::LimitsType&);
    // non blocking call to stop searching
    void stop();

    // blocking call to wait for search to finish
    void wait_for_search_finished();
    // set a new position, moves are in UCI format
    std::optional<PositionSetError> set_position(const std::string&              fen,
                                                 const std::vector<std::string>& moves);

    // modifiers

    bool set_numa_config_from_option(const std::string& o);
    void resize_threads();
    void set_tt_size(usize mb);
    void set_ponderhit(bool);
    void search_clear();

    // Freeze the TT: later searches read it but never write to or age it, keeping their own
    // results in a fresh TT of the given size. Thawing makes the frozen TT the regular one again.
    void tt_freeze(usize scratchMB);
    void tt_thaw();

    // An auxiliary search: one more thread, searching from a position of its own, with its own
    // limits and histories, and sharing only the TT with the regular search, which may keep
    // running meanwhile. It does not start a new TT generation, and its output lines start
    // with "aux".
    std::optional<PositionSetError> set_aux_position(const std::string&              fen,
                                                     const std::vector<std::string>& moves);
    void                            aux_go(Search::LimitsType&);
    void                            aux_stop();
    // The number of threads of the auxiliary search. Only waits for the auxiliary search.
    void set_aux_threads(usize n);

    void set_on_update_no_moves(std::function<void(const InfoShort&)>&&);
    void set_on_update_full(std::function<void(const InfoFull&)>&&);
    void set_on_iter(std::function<void(const InfoIter&)>&&);
    void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&);
    void set_on_start(std::function<void()>&&);
    void set_on_verify_network(std::function<void(std::string_view)>&&);
    void set_on_aux_update_no_moves(std::function<void(const InfoShort&)>&&);
    void set_on_aux_update_full(std::function<void(const InfoFull&)>&&);
    void set_on_aux_bestmove(std::function<void(std::string_view, std::string_view)>&&);

    // network related

    void                                 verify_network() const;
    std::unique_ptr<Eval::NNUE::Network> get_default_network();
    void                                 load_network(const std::filesystem::path& file);
    void save_network(const std::optional<std::filesystem::path>& file);

    // utility functions

    void trace_eval() const;

    const OptionsMap& get_options() const;
    OptionsMap&       get_options();

    int get_hashfull(int maxAge = 0) const;

    // Reports on the TT: its size and occupancy, the entry of the current position, the entries
    // of the positions after each legal move, and the PV formed by the TT moves
    std::string tt_info(bool withChecksum);
    std::string tt_probe();
    std::string tt_moves();
    std::string tt_pv(int maxPly);

    std::string                          fen() const;
    std::optional<PositionSetError>      flip();
    std::string                          visualize() const;
    std::vector<std::pair<usize, usize>> get_bound_thread_count_by_numa_node() const;
    std::string                          get_numa_config_as_string() const;
    std::string                          numa_config_information_as_string() const;
    std::string                          thread_allocation_information_as_string() const;
    std::string                          thread_binding_information_as_string() const;

   private:
    const std::filesystem::path binaryDirectory;

    NumaReplicationContext numaContext;

    Position     pos;
    StateListPtr states;

    OptionsMap                                        options;
    ThreadPool                                        threads;
    TranspositionTable                                tt;
    TranspositionTable                                frozenTT;
    Eval::NNUE::EvalFile                              networkFile;
    LazyNumaReplicatedSystemWide<Eval::NNUE::Network> network;

    Search::SearchManager::UpdateContext  updateContext;
    std::function<void(std::string_view)> onVerifyNetwork;
    std::map<NumaIndex, SharedHistories>  sharedHists;

    // the auxiliary search, declared last so that its thread is joined first
    Position                             auxPos;
    StateListPtr                         auxStates;
    Search::SearchManager::UpdateContext auxUpdateContext;
    std::map<NumaIndex, SharedHistories> auxSharedHists;
    usize                                auxThreadCount = 1;
    ThreadPool                           auxThreads;
};

}  // namespace Stockfish


#endif  // #ifndef ENGINE_H_INCLUDED
