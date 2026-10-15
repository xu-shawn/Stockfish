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

#include "engine.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <deque>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "nnue/network.h"
#include "nnue/nnue_common.h"
#include "numa.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "shm.h"
#include "syzygy/tbprobe.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

namespace NN = Eval::NNUE;

int MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

// The default configuration will attempt to group L3 domains up to 32 threads.
// This size was found to be a good balance between the Elo gain of increased
// history sharing and the speed loss from more cross-cache accesses (see
// PR#6526). The user can always explicitly override this behavior.
constexpr NumaAutoPolicy DefaultNumaPolicy = BundledL3Policy{32};

Engine::Engine(std::optional<std::filesystem::path> path) :
    binaryDirectory(path ? CommandLine::get_binary_directory(*path) : std::filesystem::path{}),
    numaContext(NumaConfig::from_system(DefaultNumaPolicy)),
    states(new std::deque<StateInfo>(1)),
    threads(),
    networkFile{std::nullopt, ""},
    network(numaContext, get_default_network()) {

    pos.set(StartFEN, false, &states->back());

    auxStates = StateListPtr(new std::deque<StateInfo>(1));
    auxPos.set(StartFEN, false, &auxStates->back());
    auxUpdateContext.onUpdateNoMoves = [](const InfoShort&) {};
    auxUpdateContext.onUpdateFull    = [](const InfoFull&) {};
    auxUpdateContext.onIter          = [](const InfoIter&) {};
    auxUpdateContext.onBestmove      = [](std::string_view, std::string_view) {};
    auxUpdateContext.onStart         = []() {};

    options.add(  //
      "Debug Log File", Option("", [](const Option& o) {
          start_logger(path_from_utf8(std::string(o)));
          return std::nullopt;
      }));

    options.add(  //
      "NumaPolicy", Option("auto", [this](const Option& o) {
          if (!set_numa_config_from_option(o))
              return "NumaPolicy: invalid value '" + std::string(o) + "', keeping previous config.";
          return numa_config_information_as_string() + "\n"
               + thread_allocation_information_as_string();
      }));

    options.add(  //
      "Threads", Option(1, 1, MaxThreads, [this](const Option&) {
          resize_threads();
          return thread_allocation_information_as_string();
      }));

    options.add(  //
      "Hash", Option(16, 1, MaxHashMB, [this](const Option& o) {
          set_tt_size(o);
          return std::nullopt;
      }));

    options.add(  //
      "Clear Hash", Option([this](const Option&) {
          search_clear();
          return std::nullopt;
      }));

    options.add(  //
      "Ponder", Option(false));

    options.add(  //
      "MultiPV", Option(1, 1, MAX_MOVES));

    options.add("Skill Level", Option(20, 0, 20));

    options.add("Move Overhead", Option(10, 0, 5000));

    options.add("nodestime", Option(0, 0, 10000));

    options.add("UCI_Chess960", Option(false));

    options.add("UCI_LimitStrength", Option(false));

    options.add("UCI_Elo",
                Option(Stockfish::Search::Skill::LowestElo, Stockfish::Search::Skill::LowestElo,
                       Stockfish::Search::Skill::HighestElo));

    options.add("UCI_ShowWDL", Option(false));

    options.add(  //
      "SyzygyPath", Option("", [](const Option& o) {
          Tablebases::init(o);
          return std::nullopt;
      }));

    options.add("SyzygyProbeDepth", Option(1, 1, 100));

    options.add("Syzygy50MoveRule", Option(true));

    options.add("SyzygyProbeLimit", Option(7, 0, 7));

    options.add(  //
      "EvalFile", Option(EvalFileDefaultName, [this](const Option& o) {
          load_network(path_from_utf8(std::string(o)));
          return std::nullopt;
      }));

    threads.clear();
    threads.ensure_network_replicated();
    resize_threads();
}

std::variant<u64, PositionSetError>
Engine::perft(const std::string& fen, Depth depth, bool isChess960) {
    verify_network();

    return Benchmark::perft(fen, depth, isChess960);
}

void Engine::go(Search::LimitsType& limits) {
    assert(limits.perft == 0);
    verify_network();

    threads.start_thinking(options, pos, states, limits);
}
void Engine::stop() { threads.stop = true; }

void Engine::search_clear() {
    wait_for_search_finished();

    tt.clear(threads);
    threads.clear();
    if (!auxThreads.empty())
        auxThreads.clear();

    // TODO: does not work with multiple instances
    Tablebases::init(options["SyzygyPath"]);  // Free mapped files
}

void Engine::tt_freeze(usize scratchMB) {
    wait_for_search_finished();

    // The frozen table takes over the current TT, freeing an earlier frozen one if there was
    // one, and searches get a fresh TT of their own
    tt.set_frozen(nullptr);
    frozenTT.swap(tt);
    tt.resize(std::max(scratchMB, usize(1)), threads);
    tt.set_frozen(&frozenTT);
}

void Engine::tt_thaw() {
    wait_for_search_finished();

    if (!tt.get_frozen())
        return;

    tt.set_frozen(nullptr);
    tt.swap(frozenTT);
    frozenTT.release();
}

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
    updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    updateContext.onBestmove = std::move(f);
}

void Engine::set_on_start(std::function<void()>&& f) { updateContext.onStart = std::move(f); }

void Engine::set_on_verify_network(std::function<void(std::string_view)>&& f) {
    onVerifyNetwork = std::move(f);
}

void Engine::set_on_aux_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    auxUpdateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_aux_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    auxUpdateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_aux_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    auxUpdateContext.onBestmove = std::move(f);
}

// Waits for both the regular and the auxiliary search, as both use the TT
void Engine::wait_for_search_finished() {
    threads.main_thread()->wait_for_search_finished();
    if (!auxThreads.empty())
        auxThreads.main_thread()->wait_for_search_finished();
}

std::optional<PositionSetError> Engine::set_aux_position(const std::string&              fen,
                                                         const std::vector<std::string>& moves) {
    // As in set_position(). A running auxiliary search owns its own copy of the states.
    auxStates = StateListPtr(new std::deque<StateInfo>(1));
    auto err  = auxPos.set(fen, options["UCI_Chess960"], &auxStates->back());

    for (usize i = 0; !err.has_value() && i < moves.size(); ++i)
    {
        auto m = UCIEngine::to_move(auxPos, moves[i]);

        if (m == Move::none())
            err = PositionSetError("Illegal move: " + moves[i]);
        else
        {
            auxStates->emplace_back();
            auxPos.do_move(m, auxStates->back());
        }
    }

    // Rather than keep a partial position, fall back to the start position
    if (err.has_value())
    {
        auxStates = StateListPtr(new std::deque<StateInfo>(1));
        auxPos.set(StartFEN, false, &auxStates->back());
    }
    return err;
}

void Engine::aux_go(Search::LimitsType& limits) {
    assert(limits.perft == 0);
    verify_network();

    if (auxThreads.empty())
    {
        auxThreads.agesTT = false;
        auxThreads.set(numaContext.get_numa_config(),
                       {options, auxThreads, tt, auxSharedHists, network}, auxUpdateContext,
                       auxThreadCount);
    }
    auxThreads.start_thinking(options, auxPos, auxStates, limits);
}

void Engine::set_aux_threads(usize n) {
    // only the auxiliary search has to be finished: the regular one may go on
    if (!auxThreads.empty())
        auxThreads.main_thread()->wait_for_search_finished();

    auxThreadCount = std::clamp(n, usize(1), usize(MaxThreads));
    if (!auxThreads.empty())
        auxThreads.set(numaContext.get_numa_config(),
                       {options, auxThreads, tt, auxSharedHists, network}, auxUpdateContext,
                       auxThreadCount);
}

void Engine::aux_stop() {
    if (!auxThreads.empty())
        auxThreads.stop = true;
}

std::optional<PositionSetError> Engine::set_position(const std::string&              fen,
                                                     const std::vector<std::string>& moves) {
    // Drop the old state and create a new one
    states   = StateListPtr(new std::deque<StateInfo>(1));
    auto err = pos.set(fen, options["UCI_Chess960"], &states->back());
    if (err.has_value())
        return err;

    for (const auto& move : moves)
    {
        auto m = UCIEngine::to_move(pos, move);

        if (m == Move::none())
            return PositionSetError("Illegal move: " + move);

        states->emplace_back();
        pos.do_move(m, states->back());
    }

    return std::nullopt;
}

// modifiers

bool Engine::set_numa_config_from_option(const std::string& o) {
    if (o == "auto" || o == "system")
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy));
    else if (o == "hardware")
        // Don't respect affinity set in the system.
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy, false));
    else if (o == "none")
        numaContext.set_numa_config(NumaConfig{});
    else
    {
        auto parsed = NumaConfig::from_string(o);
        if (!parsed.has_value())
            return false;
        numaContext.set_numa_config(std::move(*parsed));
    }

    // Force reallocation of threads in case affinities need to change.
    resize_threads();
    threads.ensure_network_replicated();
    return true;
}

void Engine::resize_threads() {
    threads.wait_for_search_finished();
    threads.set(numaContext.get_numa_config(), {options, threads, tt, sharedHists, network},
                updateContext);

    // Reallocate the hash with the new threadpool size
    set_tt_size(options["Hash"]);
    threads.ensure_network_replicated();
}

void Engine::set_tt_size(usize mb) {
    wait_for_search_finished();
    tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

// network related

void Engine::verify_network() const {
    const auto file = path_from_utf8(std::string(options["EvalFile"]));
    network->verify(onVerifyNetwork, networkFile, file);

    auto statuses = network.get_status_and_errors();
    for (usize i = 0; i < statuses.size(); ++i)
    {
        const auto [status, error] = statuses[i];
        std::string message        = "Network replica " + std::to_string(i + 1) + ": ";
        if (status == SystemWideSharedConstantAllocationStatus::NoAllocation)
            message += "No allocation.";
        else if (status == SystemWideSharedConstantAllocationStatus::LocalMemory)
            message += "Local memory.";
        else if (status == SystemWideSharedConstantAllocationStatus::SharedMemory)
            message += "Shared memory.";
        else
            message += "Unknown status.";

        if (error.has_value())
            message += " " + *error;

        onVerifyNetwork(message);
    }
}

std::unique_ptr<Eval::NNUE::Network> Engine::get_default_network() {

    auto network_ = std::make_unique<NN::Network>();

    network_->load(binaryDirectory, std::filesystem::path{}, networkFile);

    return network_;
}

void Engine::load_network(const std::filesystem::path& file) {
    network.modify_and_replicate(
      [this, &file](NN::Network& network_) { network_.load(binaryDirectory, file, networkFile); });
    threads.clear();
    threads.ensure_network_replicated();
    if (!auxThreads.empty())
        auxThreads.clear();
}

void Engine::save_network(const std::optional<std::filesystem::path>& file) {
    network.modify_and_replicate(
      [&file, this](NN::Network& network_) { network_.save(networkFile, file); });
}

// utility functions

void Engine::trace_eval() const {
    StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;
    p.set(pos.fen(), options["UCI_Chess960"], &trace_states->back());

    verify_network();

    sync_cout << "\n" << Eval::trace(p, *network) << sync_endl;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

std::optional<PositionSetError> Engine::flip() { return pos.flip(); }

std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }

namespace {

const char* bound_name(Bound b) {
    return b == BOUND_EXACT ? "exact"
         : b == BOUND_LOWER ? "lower"
         : b == BOUND_UPPER ? "upper"
                            : "none";
}

// A TT entry as text. With asMove, the entry is that of the position after a move from pos,
// and value and bound are given for the side to move in pos, i.e. as the score of the move.
std::string tt_entry(bool hit, const TTData& d, bool fromFrozen, const Position& pos, bool asMove) {
    std::stringstream ss;
    if (!hit)
        return " hit 0";

    Value v = d.value;
    Bound b = d.bound;
    if (asMove)
    {
        b = b == BOUND_LOWER ? BOUND_UPPER : b == BOUND_UPPER ? BOUND_LOWER : b;
        if (is_valid(v))
        {
            // stored values are relative to their position: one more ply to a mate from pos
            v = -v;
            v += is_win(v) ? -1 : is_loss(v) ? 1 : 0;
        }
    }
    ss << " hit 1 source " << (fromFrozen ? "frozen" : "own") << " depth " << d.depth << " bound "
       << bound_name(b) << " value "
       << (is_valid(v) ? UCIEngine::format_score(Score(v, pos)) : "none")
       << (asMove ? " reply " : " move ")
       << (d.move ? UCIEngine::move(d.move, pos.is_chess960()) : "none") << " ttpv " << d.is_pv;
    if (!asMove && is_valid(d.eval))
        ss << " eval " << UCIEngine::format_score(Score(d.eval, pos));
    return ss.str();
}

}  // namespace

// The reports below may run while a search is going on: they only read the TT, whose entries are
// relaxed atomics, and work on the position of the UCI loop, which the search does not use.
// Only the checksum, which scans all of memory, waits for the search to finish.
std::string Engine::tt_info(bool withChecksum) {
    if (withChecksum)
        wait_for_search_finished();

    std::stringstream ss;
    ss << "ttinfo size " << tt.size_mb() << " hashfull " << tt.hashfull(31);
    if (withChecksum)
        ss << " checksum " << std::hex << tt.checksum() << std::dec;
    if (tt.get_frozen())
    {
        ss << "\nttinfo frozen size " << frozenTT.size_mb() << " hashfull "
           << frozenTT.hashfull(31);
        if (withChecksum)
            ss << " checksum " << std::hex << frozenTT.checksum() << std::dec;
    }
    return ss.str();
}

std::string Engine::tt_probe() {
    auto [ttHit, ttData, ttWriter] = tt.probe(pos.key());
    return "ttprobe" + tt_entry(ttHit, ttData, ttWriter.from_frozen(), pos, false);
}

std::string Engine::tt_moves() {
    // probe the position after each move, and list the moves by their (possibly bounded) values
    std::vector<std::pair<Value, std::string>> entries;
    for (const auto& m : MoveList<ALL>(pos))
    {
        StateInfo st;
        pos.do_move(m, st);
        auto [ttHit, ttData, ttWriter] = tt.probe(pos.key());
        bool fromFrozen                = ttWriter.from_frozen();
        pos.undo_move(m);

        Value v = ttHit && is_valid(ttData.value) ? -ttData.value : -VALUE_INFINITE;
        entries.emplace_back(v, "ttmove " + UCIEngine::move(m, pos.is_chess960())
                                  + tt_entry(ttHit, ttData, fromFrozen, pos, true));
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    std::stringstream ss;
    for (const auto& e : entries)
        ss << e.second << "\n";
    ss << "ttmoves end";
    return ss.str();
}

std::string Engine::tt_pv(int maxPly) {
    std::deque<StateInfo> sts;
    std::vector<Move>     pv;
    while (int(pv.size()) < maxPly)
    {
        auto [ttHit, ttData, ttWriter] = tt.probe(pos.key());
        if (!ttHit || !ttData.move || !pos.legal(ttData.move) || pos.is_draw(int(pv.size())))
            break;
        pv.push_back(ttData.move);
        pos.do_move(ttData.move, sts.emplace_back());
    }
    for (auto it = pv.rbegin(); it != pv.rend(); ++it)
        pos.undo_move(*it);

    std::stringstream ss;
    ss << "ttpv";
    for (Move m : pv)
        ss << " " << UCIEngine::move(m, pos.is_chess960());
    return ss.str();
}

std::vector<std::pair<usize, usize>> Engine::get_bound_thread_count_by_numa_node() const {
    auto                                 counts = threads.get_bound_thread_count_by_numa_node();
    const NumaConfig&                    cfg    = numaContext.get_numa_config();
    std::vector<std::pair<usize, usize>> ratios;
    NumaIndex                            n = 0;
    for (; n < counts.size(); ++n)
        ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
    if (!counts.empty())
        for (; n < cfg.num_numa_nodes(); ++n)
            ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
    return ratios;
}

std::string Engine::get_numa_config_as_string() const {
    return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
    auto cfgStr = get_numa_config_as_string();
    return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
    auto              boundThreadsByNode = get_bound_thread_count_by_numa_node();
    std::stringstream ss;
    if (boundThreadsByNode.empty())
        return ss.str();

    bool isFirst = true;

    for (auto&& [current, total] : boundThreadsByNode)
    {
        if (!isFirst)
            ss << ":";
        ss << current << "/" << total;
        isFirst = false;
    }

    return ss.str();
}

std::string Engine::thread_allocation_information_as_string() const {
    std::stringstream ss;

    usize threadsSize = threads.size();
    ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

    auto boundThreadsByNodeStr = thread_binding_information_as_string();
    if (boundThreadsByNodeStr.empty())
        return ss.str();

    ss << " with NUMA node thread binding: ";
    ss << boundThreadsByNodeStr;

    return ss.str();
}
}
