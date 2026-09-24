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

#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "fiber.h"
#include "memory.h"
#include "misc.h"
#include "numa.h"
#include "position.h"
#include "search.h"
#include "thread_native.h"

namespace Stockfish {


class OptionsMap;
using Value = int;

// Number of search workers sharing each OS thread when searching with more than
// one thread. The workers of a thread run as fibers and hand over the thread to
// each other every few thousand nodes. A single thread keeps a single worker.
constexpr usize WorkersPerSMPThread = 2;

// Sometimes we don't want to actually bind the threads, but the recipient still
// needs to think it runs on *some* NUMA node, such that it can access structures
// that rely on NUMA node knowledge. This class encapsulates this optional process
// such that the recipient does not need to know whether the binding happened or not.
class OptionalThreadToNumaNodeBinder {
   public:
    OptionalThreadToNumaNodeBinder(NumaIndex n) :
        numaConfig(nullptr),
        numaId(n) {}

    OptionalThreadToNumaNodeBinder(const NumaConfig& cfg, NumaIndex n) :
        numaConfig(&cfg),
        numaId(n) {}

    NumaReplicatedAccessToken operator()() const {
        if (numaConfig != nullptr)
            return numaConfig->bind_current_thread_to_numa_node(numaId);
        else
            return NumaReplicatedAccessToken(numaId);
    }

   private:
    const NumaConfig* numaConfig;
    NumaIndex         numaId;
};

// Abstraction of a thread. It contains one or more workers and a native thread.
// After construction, the native thread is started with idle_loop()
// waiting for a signal to start searching.
// When the signal is received, the thread starts searching and when
// the search is finished, it goes back to idle_loop() waiting for a new signal.
// If there are several workers, they run as fibers on the native thread.
class Thread {
   public:
    Thread(Search::SharedState&,
           std::unique_ptr<Search::SearchManager>,
           usize,
           usize,
           usize,
           usize,
           OptionalThreadToNumaNodeBinder);
    virtual ~Thread();

    void idle_loop();
    void start_searching();
    void clear_workers();

    // Used by the main worker to start the other workers sharing the main
    // thread. Must be called from within the main worker's fiber.
    void start_main_thread_helpers();

    void run_custom_job(std::function<void()> f);

    void ensure_network_replicated();

    // Thread has been slightly altered to allow running custom jobs, so
    // this name is no longer correct. However, this class (and ThreadPool)
    // require further work to make them properly generic while maintaining
    // appropriate specificity regarding search, from the point of view of an
    // outside user, so renaming of this function is left for whenever that happens.
    void  wait_for_search_finished();
    usize id() const { return idx; }

    std::vector<LargePagePtr<Search::Worker>> workers;
    std::function<void()>                     jobFunc;

   private:
    std::mutex                mutex;
    std::condition_variable   cv;
    usize                     idx, idxInNuma, totalNuma, workerCount;
    bool                      exit = false, searching = true;  // Set before starting std::thread
    NativeThread              stdThread;
    NumaReplicatedAccessToken numaAccessToken;
    std::unique_ptr<FiberScheduler> scheduler;

    void start_fiber(usize workerIdx);
};


// ThreadPool struct handles all the threads-related stuff like init, starting,
// parking and, most importantly, launching a thread. All the access to threads
// is done through this class.
class ThreadPool {
   public:
    ThreadPool() {}

    ~ThreadPool() {
        // destroy any existing thread(s)
        if (threads.size() > 0)
        {
            main_thread()->wait_for_search_finished();

            threads.clear();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&)      = delete;

    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&)      = delete;

    void  start_thinking(const OptionsMap&, Position&, StateListPtr&, Search::LimitsType);
    void  run_on_thread(usize threadId, std::function<void()> f);
    void  wait_on_thread(usize threadId);
    usize num_threads() const;
    void  clear();
    void  set(const NumaConfig& numaConfig,
              Search::SharedState,
              const Search::SearchManager::UpdateContext&);

    Search::SearchManager* main_manager();
    Thread*                main_thread() const { return threads.front().get(); }
    u64                    nodes_searched() const;
    u64                    tb_hits() const;
    Search::Worker*        get_best_worker() const;
    void                   start_searching();
    void                   wait_for_search_finished() const;

    std::vector<usize> get_bound_thread_to_numa_node() const;
    std::vector<usize> get_bound_thread_count_by_numa_node() const;
    usize              numa_nodes() const;

    void ensure_network_replicated();

    std::atomic_bool stop, increaseDepth;

    auto cbegin() const noexcept { return threads.cbegin(); }
    auto begin() noexcept { return threads.begin(); }
    auto end() noexcept { return threads.end(); }
    auto cend() const noexcept { return threads.cend(); }
    auto size() const noexcept { return threads.size(); }
    auto empty() const noexcept { return threads.empty(); }

    const std::vector<Search::Worker*>& all_workers() const { return workers; }

   private:
    StateListPtr                         setupStates;
    std::vector<std::unique_ptr<Thread>> threads;
    std::vector<Search::Worker*>         workers;  // All workers of all threads, by index
    std::vector<NumaIndex>               boundThreadToNumaNode;

    u64 accumulate(RelaxedAtomic<u64> Search::Worker::* member) const {

        u64 sum = 0;
        for (Search::Worker* w : workers)
            sum += (w->*member).load(std::memory_order_relaxed);
        return sum;
    }
};

}  // namespace Stockfish

#endif  // #ifndef THREAD_H_INCLUDED
