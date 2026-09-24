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

// The ucontext routines are not implemented on arm64 macOS. Fail the build rather
// than the bench.
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    #error "Fibers are not supported on Apple Silicon"
#endif

// The deprecated ucontext routines are only declared with _XOPEN_SOURCE on macOS
#if defined(__APPLE__) && !defined(_XOPEN_SOURCE)
    #define _XOPEN_SOURCE 700
    #define _DARWIN_C_SOURCE
#endif

#include "fiber.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <ucontext.h>
    #include <unistd.h>
#endif

namespace Stockfish {

namespace {

// Same as the stack size of the search threads, see thread_native.h
constexpr usize FiberStackSize = 8 * 1024 * 1024;

[[noreturn]] void fiber_error(const char* msg) {
    std::cerr << "Fiber error: " << msg << std::endl;
    std::exit(EXIT_FAILURE);
}

}  // namespace

struct FiberScheduler::Context {
#if defined(_WIN32)
    void* fiber = nullptr;
#else
    ucontext_t uc;
#endif
};

namespace {

void switch_context(FiberScheduler::Context& from, FiberScheduler::Context& to) {
#if defined(_WIN32)
    (void) from;
    SwitchToFiber(to.fiber);
#else
    swapcontext(&from.uc, &to.uc);
#endif
}

}  // namespace

struct FiberScheduler::Fiber {
    explicit Fiber(FiberScheduler& s);
    ~Fiber();

    // Fibers are reused for all tasks, so this never returns: after a task has
    // finished the fiber is parked here until the scheduler assigns a new one.
    [[noreturn]] void main_loop() {
        while (true)
        {
            task();
            task    = nullptr;
            running = false;
            --owner.active;
            switch_context(context, *owner.schedulerContext);
        }
    }

    FiberScheduler&       owner;
    std::function<void()> task;
    bool                  running = false;
    Context               context;

#if !defined(_WIN32)
    void* stack;
    usize mappedSize;
#endif
};

namespace {

#if defined(_WIN32)

VOID WINAPI fiber_entry(LPVOID param) { static_cast<FiberScheduler::Fiber*>(param)->main_loop(); }

#else

// makecontext() can only pass int arguments, so the pointer is split in two halves
void fiber_entry(unsigned hi, unsigned lo) {
    auto p = std::uintptr_t((u64(hi) << 32) | lo);
    reinterpret_cast<FiberScheduler::Fiber*>(p)->main_loop();
}

#endif

}  // namespace

#if defined(_WIN32)

FiberScheduler::Fiber::Fiber(FiberScheduler& s) :
    owner(s) {
    context.fiber = CreateFiberEx(0, FiberStackSize, FIBER_FLAG_FLOAT_SWITCH, fiber_entry, this);
    if (!context.fiber)
        fiber_error("CreateFiberEx() failed");
}

FiberScheduler::Fiber::~Fiber() { DeleteFiber(context.fiber); }

#else

FiberScheduler::Fiber::Fiber(FiberScheduler& s) :
    owner(s) {

    // Reserve an extra guard page below the stack to catch overflows
    const usize pageSize = usize(sysconf(_SC_PAGESIZE));
    mappedSize           = FiberStackSize + pageSize;
    stack = mmap(nullptr, mappedSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (stack == MAP_FAILED)
        fiber_error("mmap() of the fiber stack failed");

    mprotect(stack, pageSize, PROT_NONE);

    if (getcontext(&context.uc) != 0)
        fiber_error("getcontext() failed, ucontext is not supported on this platform");

    context.uc.uc_stack.ss_sp   = static_cast<char*>(stack) + pageSize;
    context.uc.uc_stack.ss_size = FiberStackSize;
    context.uc.uc_link          = nullptr;

    const u64 p = u64(reinterpret_cast<std::uintptr_t>(this));
    makecontext(&context.uc, reinterpret_cast<void (*)()>(&fiber_entry), 2, unsigned(p >> 32),
                unsigned(p));
}

FiberScheduler::Fiber::~Fiber() { munmap(stack, mappedSize); }

#endif

FiberScheduler::FiberScheduler(usize count) :
    schedulerContext(std::make_unique<Context>()) {
    for (usize i = 0; i < count; ++i)
        fibers.push_back(std::make_unique<Fiber>(*this));
}

FiberScheduler::~FiberScheduler() { assert(!active); }

void FiberScheduler::start(usize i, std::function<void()> task) {
    assert(i < fibers.size() && !fibers[i]->running);

    fibers[i]->task    = std::move(task);
    fibers[i]->running = true;
    ++active;
}

void FiberScheduler::run() {

#if defined(_WIN32)
    // The OS thread must be converted into a fiber before it can switch to others
    void* self              = ConvertThreadToFiber(nullptr);
    bool  converted         = self != nullptr;
    schedulerContext->fiber = converted ? self : GetCurrentFiber();
#endif

    while (active)
        for (current = 0; current < fibers.size(); ++current)
            if (fibers[current]->running)
                switch_context(*schedulerContext, fibers[current]->context);

#if defined(_WIN32)
    if (converted)
        ConvertFiberToThread();
#endif
}

void FiberScheduler::yield() {

    // The calling fiber is the only one left
    if (active < 2)
        return;

    switch_context(fibers[current]->context, *schedulerContext);
}

void FiberScheduler::wait_for_others() {
    while (active > 1)
        yield();
}

}  // namespace Stockfish
