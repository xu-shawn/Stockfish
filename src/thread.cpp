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

#include "thread.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "bitboard.h"
#include "history.h"
#include "memory.h"
#include "movegen.h"
#include "search.h"
#include "syzygy/tbprobe.h"
#include "timeman.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

// Constructor launches the thread and waits until it goes to sleep
// in idle_loop(). Note that 'searching' and 'exit' should be already set.
Thread::Thread(Search::SharedState&                   sharedState,
               std::unique_ptr<Search::SearchManager> sm,
               usize                                  n,
               usize                                  numaN,
               usize                                  totalNumaCount,
               usize                                  workersPerThread,
               OptionalThreadToNumaNodeBinder         binder) :
    idx(n),
    idxInNuma(numaN),
    totalNuma(totalNumaCount),
    workerCount(workersPerThread),
    stdThread(
      create_native_thread(NativeThreadOptions{}.setLargeStack(true), &Thread::idle_loop, this)) {

    if (!stdThread.joinable())
    {
        std::cerr << "Failed to create search thread\n";
        std::exit(EXIT_FAILURE);
    }

    wait_for_search_finished();

    run_custom_job([this, &binder, &sharedState, &sm, n]() {
        // Use the binder to [maybe] bind the threads to a NUMA node before doing
        // the Worker allocation. Ideally we would also allocate the SearchManager
        // here, but that's minor.
        this->numaAccessToken = binder();

        // Workers are indexed as if each of them had its own thread. Only the
        // first worker of the main thread gets the SearchManager.
        for (usize k = 0; k < workerCount; ++k)
            this->workers.push_back(make_unique_large_page<Search::Worker>(
              sharedState, k == 0 ? std::move(sm) : nullptr, n * workerCount + k,
              idxInNuma * workerCount + k, totalNuma * workerCount, this->numaAccessToken));

        if (workerCount > 1)
        {
            this->scheduler = std::make_unique<FiberScheduler>(workerCount);
            for (auto&& w : this->workers)
                w->scheduler = this->scheduler.get();
        }
    });

    wait_for_search_finished();
}


// Destructor wakes up the thread in idle_loop() and waits
// for its termination. Thread should be already waiting.
Thread::~Thread() {

    assert(!searching);

    exit = true;
    start_searching();
    stdThread.join();
}

// Wakes up the thread that will start the search. On the main thread only the
// main worker is started here, it starts the other workers of the main thread
// through start_main_thread_helpers() once the search is set up.
void Thread::start_searching() {
    assert(!workers.empty());
    run_custom_job([this]() {
        if (!scheduler)
        {
            workers.front()->start_searching();
            return;
        }

        for (usize k = 0; k < (idx == 0 ? 1 : workerCount); ++k)
            start_fiber(k);

        scheduler->run();
    });
}

void Thread::start_fiber(usize workerIdx) {
    Search::Worker* w = workers[workerIdx].get();
    scheduler->start(workerIdx, [w]() { w->start_searching(); });
}

void Thread::start_main_thread_helpers() {
    assert(idx == 0);
    for (usize k = 1; k < workerCount; ++k)
        start_fiber(k);
}

// Clears the histories for the thread workers (usually before a new game)
void Thread::clear_workers() {
    assert(!workers.empty());
    run_custom_job([this]() {
        for (auto&& w : workers)
            w->clear();
    });
}

// Blocks on the condition variable until the thread has finished searching
void Thread::wait_for_search_finished() {

    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [&] { return !searching; });
}

// Launching a function in the thread
void Thread::run_custom_job(std::function<void()> f) {
    {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [&] { return !searching; });
        jobFunc   = std::move(f);
        searching = true;
    }
    cv.notify_one();
}

void Thread::ensure_network_replicated() {
    for (auto&& w : workers)
        w->ensure_network_replicated();
}

// Thread gets parked here, blocked on the condition variable
// when the thread has no work to do.

void Thread::idle_loop() {
    while (true)
    {
        std::unique_lock<std::mutex> lk(mutex);
        searching = false;
        cv.notify_one();  // Wake up anyone waiting for search finished
        cv.wait(lk, [&] { return searching; });

        if (exit)
            return;

        std::function<void()> job = std::move(jobFunc);
        jobFunc                   = nullptr;

        lk.unlock();

        if (job)
            job();
    }
}

Search::SearchManager* ThreadPool::main_manager() { return workers.front()->main_manager(); }

u64 ThreadPool::nodes_searched() const { return accumulate(&Search::Worker::nodes); }
u64 ThreadPool::tb_hits() const { return accumulate(&Search::Worker::tbHits); }

static usize next_power_of_two(u64 count) { return count > 1 ? (2ULL << msb(count - 1)) : 1; }

// Creates/destroys threads to match the requested number.
// Created and launched threads will immediately go to sleep in idle_loop.
// Upon resizing, threads are recreated to allow for binding if necessary.
void ThreadPool::set(const NumaConfig&                           numaConfig,
                     Search::SharedState                         sharedState,
                     const Search::SearchManager::UpdateContext& updateContext) {

    if (threads.size() > 0)  // destroy any existing thread(s)
    {
        main_thread()->wait_for_search_finished();

        workers.clear();
        threads.clear();

        boundThreadToNumaNode.clear();
    }

    const usize requested        = sharedState.options["Threads"];
    const usize workersPerThread = requested > 1 ? WorkersPerSMPThread : 1;

    if (requested > 0)  // create new thread(s)
    {
        // Binding threads may be problematic when there's multiple NUMA nodes and
        // multiple Stockfish instances running. In particular, if each instance
        // runs a single thread then they would all be mapped to the first NUMA node.
        // This is undesirable, and so the default behaviour (i.e. when the user does not
        // change the NumaConfig UCI setting) is to not bind the threads to processors
        // unless we know for sure that we span NUMA nodes and replication is required.
        const std::string numaPolicy(sharedState.options["NumaPolicy"]);
        const bool        doBindThreads = [&]() {
            if (numaPolicy == "none")
                return false;

            if (numaPolicy == "auto")
                return numaConfig.suggests_binding_threads(requested);

            // numaPolicy == "system", or explicitly set by the user
            return true;
        }();

        std::map<NumaIndex, usize> counts;
        boundThreadToNumaNode = doBindThreads
                                ? numaConfig.distribute_threads_among_numa_nodes(requested)
                                : std::vector<NumaIndex>{};

        if (boundThreadToNumaNode.empty())
            counts[0] = requested;  // Pretend all threads are part of numa node 0
        else
        {
            for (usize i = 0; i < boundThreadToNumaNode.size(); ++i)
                counts[boundThreadToNumaNode[i]]++;
        }

        sharedState.sharedHistories.clear();
        for (auto pair : counts)
        {
            NumaIndex numaIndex = pair.first;
            u64       count     = pair.second;
            auto      f         = [&]() {
                sharedState.sharedHistories.try_emplace(
                  numaIndex, next_power_of_two(count * workersPerThread));
            };
            if (doBindThreads)
                numaConfig.execute_on_numa_node(numaIndex, f);
            else
                f();
        }

        auto threadsPerNode = counts;
        counts.clear();

        while (threads.size() < requested)
        {
            const usize     threadId      = threads.size();
            const NumaIndex numaId        = doBindThreads ? boundThreadToNumaNode[threadId] : 0;
            auto            create_thread = [&]() {
                auto manager =
                  threadId == 0 ? std::make_unique<Search::SearchManager>(updateContext) : nullptr;

                // When not binding threads we want to force all access to happen
                // from the same NUMA node, because in case of NUMA replicated memory
                // accesses we don't want to trash cache in case the threads get scheduled
                // on the same NUMA node.
                auto binder = doBindThreads ? OptionalThreadToNumaNodeBinder(numaConfig, numaId)
                                                       : OptionalThreadToNumaNodeBinder(numaId);

                threads.emplace_back(std::make_unique<Thread>(
                  sharedState, std::move(manager), threadId, counts[numaId]++,
                  threadsPerNode[numaId], workersPerThread, binder));
            };

            // Ensure the worker thread inherits the intended NUMA affinity at creation.
            if (doBindThreads)
                numaConfig.execute_on_numa_node(numaId, create_thread);
            else
                create_thread();
        }

        for (auto&& th : threads)
            for (auto&& w : th->workers)
                workers.push_back(w.get());

        clear();

        main_thread()->wait_for_search_finished();
    }
}


// Sets threadPool data to initial values
void ThreadPool::clear() {
    if (threads.empty())
        return;

    for (auto&& th : threads)
        th->clear_workers();

    for (auto&& th : threads)
        th->wait_for_search_finished();

    // These two affect the time taken on the first move of a game:
    main_manager()->bestPreviousAverageScore = VALUE_INFINITE;
    main_manager()->previousTimeReduction    = 0.85;

    main_manager()->callsCnt           = 0;
    main_manager()->bestPreviousScore  = VALUE_INFINITE;
    main_manager()->originalTimeAdjust = -1;
    main_manager()->tm.clear();
}

void ThreadPool::run_on_thread(usize threadId, std::function<void()> f) {
    assert(threads.size() > threadId);
    threads[threadId]->run_custom_job(std::move(f));
}

void ThreadPool::wait_on_thread(usize threadId) {
    assert(threads.size() > threadId);
    threads[threadId]->wait_for_search_finished();
}

usize ThreadPool::num_threads() const { return threads.size(); }


// Wakes up main thread waiting in idle_loop() and returns immediately.
// Main thread will wake up other threads and start the search.
void ThreadPool::start_thinking(const OptionsMap&  options,
                                Position&          pos,
                                StateListPtr&      states,
                                Search::LimitsType limits) {

    main_thread()->wait_for_search_finished();

    main_manager()->stopOnPonderhit = stop = false;
    main_manager()->ponder                 = limits.ponderMode;

    increaseDepth = true;

    Search::RootMoves rootMoves;

    for (const auto& uciMove : limits.searchmoves)
    {
        auto move = UCIEngine::to_move(pos, uciMove);

        if (move != Move::none())
            rootMoves.emplace_back(move);
    }

    if (rootMoves.empty())
        for (const auto& m : MoveList<LEGAL>(pos))
            rootMoves.emplace_back(m);

    Tablebases::Config tbConfig = Tablebases::rank_root_moves(options, pos, rootMoves);

    // After ownership transfer 'states' becomes empty, so if we stop the search
    // and call 'go' again without setting a new position states.get() == nullptr.
    assert(states.get() || setupStates.get());

    if (states.get())
        setupStates = std::move(states);  // Ownership transfer, states is now empty

    // We use Position::set() to set root position across threads. But there are
    // some StateInfo fields (previous, pliesFromNull, capturedPiece) that cannot
    // be deduced from a fen string, so set() clears them and they are set from
    // setupStates->back() later. The rootState is per thread, earlier states are
    // shared since they are read-only.
    for (auto&& th : threads)
    {
        th->run_custom_job([&]() {
            for (auto&& w : th->workers)
            {
                w->limits = limits;
                w->nodes = w->tbHits = w->bestMoveChanges = 0;
                w->nmpMinPly                              = 0;
                w->rootDepth                              = 0;
                w->rootMoves                              = rootMoves;
                w->rootPos.set(pos.fen(), pos.is_chess960(), &w->rootState);
                w->rootState = setupStates->back();
                w->tbConfig  = tbConfig;
            }
        });
    }

    for (auto&& th : threads)
        th->wait_for_search_finished();

    main_thread()->start_searching();
}

Search::Worker* ThreadPool::get_best_worker() const {

    Search::Worker* bestThread = workers.front();
    Value           minScore   = VALUE_INFINITE;

    std::unordered_map<Move, i64, Move::MoveHash> votes(
      2 * std::min(workers.size(), bestThread->rootMoves.size()));

    for (Search::Worker* th : workers)
        minScore = std::min(minScore, th->rootMoves[0].score);

    // Vote according to score, and select the best worker
    for (Search::Worker* th : workers)
        votes[th->rootMoves[0].pv[0]] += th->rootMoves[0].score - minScore + 14;

    for (Search::Worker* th : workers)
    {
        const auto& bestThreadMove = bestThread->rootMoves[0];
        const auto& newThreadMove  = th->rootMoves[0];

        const auto bestThreadMoveVote = votes[bestThreadMove.pv[0]];
        const auto newThreadMoveVote  = votes[newThreadMove.pv[0]];

        // Aborted (d1) searches may lead to inexact win (or loss) scores.
        const bool bestThreadDecisive = bestThreadMove.score != -VALUE_INFINITE
                                     && is_decisive(bestThreadMove.score)
                                     && !bestThreadMove.is_inexact();
        const bool newThreadDecisive = newThreadMove.score != -VALUE_INFINITE
                                    && is_decisive(newThreadMove.score)
                                    && !newThreadMove.is_inexact();

        if (bestThreadDecisive)
        {
            // Make sure we pick the shortest mate / TB conversion.
            if (newThreadDecisive && std::abs(newThreadMove.score) > std::abs(bestThreadMove.score))
            {
                assert((is_win(bestThreadMove.score) && is_win(newThreadMove.score))
                       || (is_loss(bestThreadMove.score) && is_loss(newThreadMove.score)));

                bestThread = th;
            }
        }
        else if (newThreadDecisive
                 || (!is_loss(newThreadMove.score)
                     && (newThreadMoveVote > bestThreadMoveVote
                         || (newThreadMoveVote == bestThreadMoveVote
                             && newThreadMove.pv.size() > bestThreadMove.pv.size()))))
            bestThread = th;
    }

    return bestThread;
}


// Start non-main workers, including the ones sharing the main thread.
// Will be invoked by main worker after it has started searching.
void ThreadPool::start_searching() {

    for (auto&& th : threads)
        if (th != threads.front())
            th->start_searching();

    main_thread()->start_main_thread_helpers();
}


// Wait for non-main threads. The main worker waits for the other workers
// sharing its thread separately, see Search::Worker::start_searching().
void ThreadPool::wait_for_search_finished() const {

    for (auto&& th : threads)
        if (th != threads.front())
            th->wait_for_search_finished();
}

std::vector<usize> ThreadPool::get_bound_thread_to_numa_node() const {
    return boundThreadToNumaNode;
}

std::vector<usize> ThreadPool::get_bound_thread_count_by_numa_node() const {
    std::vector<usize> counts;

    if (!boundThreadToNumaNode.empty())
    {
        NumaIndex highestNumaNode = 0;
        for (NumaIndex n : boundThreadToNumaNode)
            if (n > highestNumaNode)
                highestNumaNode = n;

        counts.resize(highestNumaNode + 1, 0);

        for (NumaIndex n : boundThreadToNumaNode)
            counts[n] += 1;
    }

    return counts;
}

usize ThreadPool::numa_nodes() const {
    std::unordered_set<usize> seen;
    for (NumaIndex n : boundThreadToNumaNode)
        seen.insert(n);
    return std::max(seen.size(), usize(1));
}

void ThreadPool::ensure_network_replicated() {
    for (auto&& th : threads)
        th->ensure_network_replicated();
}

}  // namespace Stockfish
