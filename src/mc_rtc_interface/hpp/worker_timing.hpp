#pragma once
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <unistd.h>

class WorkerTiming
{
    public:
        using clock = std::chrono::steady_clock;

        WorkerTiming()
        {
            const std::string path = "/tmp/mc_mjlab_timing_" + std::to_string(::getpid()) + ".log";
            m_file                 = std::fopen(path.c_str(), "a");
        }
        ~WorkerTiming() { if (m_file) std::fclose(m_file); }

        // One whole Step period. Construct on the stack at the top of the step branch.
        struct Period
        {
            WorkerTiming &t; clock::time_point t0 = clock::now();
            explicit Period(WorkerTiming &timing) : t(timing) {}
            ~Period() { t.add_period(std::chrono::duration<double, std::milli>(clock::now() - t0).count()); }
        };

        // One row inside a period.
        struct Row
        {
            WorkerTiming &t; clock::time_point t0 = clock::now(); bool reset = false;
            explicit Row(WorkerTiming &timing) : t(timing) {}
            ~Row() { t.add_row(std::chrono::duration<double, std::milli>(clock::now() - t0).count(), reset); }
        };

    private:
        void add_row(double ms, bool reset)
        {
            if (ms > m_row_max) m_row_max = ms;
            if (reset) ++m_resets;
        }
        void add_period(double ms)
        {
            m_sum += ms; m_row_max_sum += m_row_max; m_row_max = 0.0;
            if (ms > m_max) m_max = ms;
            if (++m_n == window)
            {
                if (m_file)
                {
                    std::fprintf(m_file,
                        "periods=%zu mean=%.2fms max=%.2fms mean_slowest_row=%.2fms resets=%zu\n",
                        m_n, m_sum / m_n, m_max, m_row_max_sum / m_n, m_resets);
                    std::fflush(m_file);
                }
                m_n = 0; m_sum = m_max = m_row_max_sum = 0.0; m_resets = 0;
            }
        }

        static constexpr std::size_t window = 100;
        FILE       *m_file = nullptr;
        double      m_sum = 0.0, m_max = 0.0, m_row_max = 0.0, m_row_max_sum = 0.0;
        std::size_t m_n = 0, m_resets = 0;
};