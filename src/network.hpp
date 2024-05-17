/*
music-detect
Copyright (C) 2024 Alex <uni@vrsal.xyz>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <unordered_map>

#include "inference.hpp"

class Network {
    torch::jit::Module m_model {};
    bool m_loaded = false;

    std::thread m_inference_thread;
    std::mutex m_mutex;
    bool m_thread_running = true;
    int m_job_counter = 0;
    std::deque<std::pair<int, torch::Tensor>> m_inference_queue;
    std::unordered_map<int, InferenceResult> m_inference_results;

    void inference_thread_method();

public:
    Network();
    ~Network();

    void load();
    int run(float* frames, int count);

    bool get_result(int job_id, InferenceResult& result);
};

extern Network* g_network;

