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

#include "network.hpp"

#include <obs-module.h>
#include <obs/util/util.hpp>

#include "labels.hpp"
#include "log.h"

Network* g_network = nullptr;

Network::Network() { }

Network::~Network()
{
    m_mutex.lock();
    m_thread_running = false;
    m_mutex.unlock();
    m_inference_thread.join();
}

void Network::inference_thread_method()
{
    while (true) {
        m_mutex.lock();
        if (!m_thread_running) {
            m_mutex.unlock();
            break;
        }

        if (m_inference_queue.empty()) {
            m_mutex.unlock();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(100));
            continue;
        }

        auto [job_id, data] = m_inference_queue.front();
        m_inference_queue.pop_front();
        m_mutex.unlock();

        try {
            auto outputs = m_model.forward({ data }).toTuple();
            auto output = outputs->elements().at(0).toTensor().cpu().contiguous().to(torch::kFloat32)[0];
            auto output_a = output.accessor<float, 1>();
            auto sorted = output.sort(0);

            auto idx = std::get<1>(sorted);
            idx = idx.flip({ 0 });
            auto i_a = idx.accessor<long, 1>();

            InferenceResult result;
            for (int i = 0; i < 10; i++) {
                result.labels[i] = g_lables[(size_t)i_a[i]].c_str();
                result.confidences[i] = output_a[i_a[i]];
            }

            m_inference_results[job_id] = result;
        } catch (const std::exception& e) {
            berr("Failed to run inference: %s", e.what());
        }
    }
}

void Network::load()
{
    BPtr<char> path = obs_module_file("CNN6.pt");
    try {
        m_model = torch::jit::load(std::string(path));
    } catch (const std::exception& e) {
        berr("Failed to get path to model file: %s", e.what());
        return;
    }
    binfo("Model loaded successfully");
    m_loaded = true;
    m_inference_thread = std::thread(std::bind(&Network::inference_thread_method, this));
}

int Network::run(float* frames, int count)
{
    if (!m_loaded) {
        berr("Model not loaded");
        return -1;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    torch::Tensor tensor = torch::from_blob(frames, { count }, torch::kFloat32).clone();
    tensor = tensor.to(torch::kCPU);
    tensor = tensor.reshape({ 1, count });

    torch::Tensor mask = torch::isnan(tensor);
    tensor = torch::where(mask, torch::zeros_like(tensor), tensor);

    int job_id = m_job_counter++;
    m_inference_queue.push_back({ job_id, tensor });
    return job_id;
}

bool Network::get_result(int job_id, InferenceResult& result)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_inference_results.find(job_id);
    if (it == m_inference_results.end()) {
        return false;
    }
    result = it->second;
    return true;
}
