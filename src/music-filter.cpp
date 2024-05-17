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
#include "music-filter.hpp"

#if defined(__APPLE__) || defined(_WIN32)
#include <media-io/audio-resampler.h>
#include <util/deque.h>
#include <util/platform.h>
#else
#include <obs/media-io/audio-resampler.h>
#include <obs/util/deque.h>
#include <obs/util/platform.h>
#endif

#include <string>

#include "inference.hpp"
#include "log.h"

#define T_(s)                   obs_module_text(s)
#define T_CLASS_TO_IDENTIFY     T_("targetclass")
#define T_WINDOW_SIZE           T_("windowsize")
#define T_CONFIDENCE_THRESHOLD  T_("confidencethreshold")
#define T_SMOOTHING             T_("smoothing")
#define T_CONFIDENCE_INFO       T_("confidenceinfo")
#define T_LOG                   T_("log")

#define S_WINDOW_SIZE           "window_size"
#define S_CLASS_TO_IDENTIFY     "target_class"
#define S_CONFIDENCE_THRESHOLD  "confidence_threshold"
#define S_SMOOTHING             "smoothing"
#define S_LOG                   "log"

struct music_filter_data {
    obs_source_t* context{};
    audio_resampler_t* resampler{};
    struct deque buffer{};
    bool valid{};
    uint64_t last_submit_time{};
    int current_inference_job_id{};
    uint64_t analyze_window_seconds{};
    std::string target_class{};
    float threshold{};
    float smoothing{};
    float new_confidence{};
    float current_confidence{};
    bool log_confidence {};
};

static const char* mf_name(void*)
{
    return "Music filter";
}

static void mf_destroy(void* data)
{
    auto filter = static_cast<music_filter_data*>(data);
    if (filter->resampler) {
        audio_resampler_destroy(filter->resampler);
    }
    deque_free(&filter->buffer);
    delete filter;
}

static void mf_update(void* data, obs_data_t* s)
{
    auto filter = static_cast<music_filter_data*>(data);

    filter->analyze_window_seconds = (uint64_t) obs_data_get_int(s, S_WINDOW_SIZE);
    filter->target_class = obs_data_get_string(s, S_CLASS_TO_IDENTIFY);
    filter->threshold = (float) obs_data_get_double(s, S_CONFIDENCE_THRESHOLD);
    filter->smoothing = (float) obs_data_get_double(s, S_SMOOTHING);
    filter->log_confidence = obs_data_get_bool(s, S_LOG);
}

static void* mf_create(obs_data_t* settings, obs_source_t* filter)
{
    UNUSED_PARAMETER(settings);
    auto* mfd = new music_filter_data;
    mfd->context = filter;
    mfd->current_inference_job_id = -1;
    mfd->analyze_window_seconds = 2;
    deque_init(&mfd->buffer);
    // make room for 5 seconds of mono float audio at 32000 samples per second
    deque_reserve(&mfd->buffer, 32000 * mfd->analyze_window_seconds * sizeof(float));

    struct obs_audio_info info = {};
    if (obs_get_audio_info(&info)) {
        mfd->valid = true;
        struct resample_info dst = {
            .samples_per_sec = 32000, // CNN6 model uses this
            .format = AUDIO_FORMAT_FLOAT_PLANAR,
            .speakers = SPEAKERS_MONO
        };
        struct resample_info src = {
            .samples_per_sec = info.samples_per_sec,
            .format = AUDIO_FORMAT_FLOAT_PLANAR, // I think obs always uses float
            .speakers = info.speakers
        };
        mfd->resampler = audio_resampler_create(&dst, &src);
    } else {
        blog(LOG_ERROR, "Failed to get audio info");
    }
    mf_update(mfd, settings);
    return mfd;
}

static struct obs_audio_data* mf_filter_audio(void* data,
    struct obs_audio_data* audio)
{
    auto filter = static_cast<music_filter_data*>(data);
    if (!filter->valid) {
        return audio;
    }

    // low-pass filter for confidences smoothing
    filter->current_confidence = filter->current_confidence * (1.f - filter->smoothing) + filter->new_confidence * filter->smoothing;
    auto* src = obs_filter_get_target(filter->context);
    if (src)
        obs_source_set_muted(src, filter->current_confidence > filter->threshold);

    if (audio->frames <= 0) {
        return audio;
    }

    uint64_t cur_time = os_gettime_ns();
    if (filter->last_submit_time == 0) {
        filter->last_submit_time = cur_time;
    }

    uint8_t *out[MAX_AV_PLANES];
    uint32_t out_frames = 0;
    uint64_t out_ts_offset = 0;
    bool success;

    success = audio_resampler_resample(filter->resampler, out, &out_frames,
        &out_ts_offset, (const uint8_t* const*)audio->data, audio->frames);

    // Push audio data to buffer
    if (success)
        deque_push_back(&filter->buffer, out[0], out_frames * sizeof(float));

    if (filter->current_inference_job_id == -1) {
        // After 5 seconds, resample the buffer and submit it to the inference engine
        if (cur_time - filter->last_submit_time >= 1000000000 * filter->analyze_window_seconds) {
            filter->last_submit_time = cur_time;

            filter->current_inference_job_id = run_inference((float*)filter->buffer.data, static_cast<int>(filter->buffer.size / sizeof(float)));
            deque_pop_front(&filter->buffer, NULL, filter->buffer.size);
        }
    } else {
        struct InferenceResult result;

        if (get_inference_result(filter->current_inference_job_id,
                &result)) {
            filter->current_inference_job_id = -1;
            if (filter->log_confidence) {
                binfo("Current confidence: %f, new confidence: %f", filter->current_confidence, filter->new_confidence);
                binfo("==== results ====");
            }
            for (int i = 0; i < 10; i++) {
                if (result.labels[i] == filter->target_class)
                    filter->new_confidence = result.confidences[i];
                if (filter->log_confidence) {
                    binfo(" - %s: %f", result.labels[i], result.confidences[i]);
                }
            }

        }
    }
    return audio;
}

static obs_properties_t* mf_properties(void* data)
{
    UNUSED_PARAMETER(data);
    auto* props = obs_properties_create();

    obs_properties_add_text(props, S_CLASS_TO_IDENTIFY, T_CLASS_TO_IDENTIFY, OBS_TEXT_DEFAULT);
    obs_properties_add_text(props, "infotext", "<a href=\"https://github.com/univrsal/music-detect/blob/master/src/labels.cpp\">List of classes</a>", OBS_TEXT_INFO);

    auto threshold = obs_properties_add_float(props, S_CONFIDENCE_THRESHOLD, T_CONFIDENCE_THRESHOLD, 0.01, 1.0, 0.01);
    obs_property_set_long_description(threshold, T_CONFIDENCE_INFO);

    obs_properties_add_float(props, S_SMOOTHING, T_SMOOTHING, 0.0, 0.9, 0.01);
    auto size = obs_properties_add_int(props, S_WINDOW_SIZE, T_WINDOW_SIZE, 1, 30, 1);

    obs_property_int_set_suffix(size, " seconds");
    obs_property_int_set_suffix(threshold, " %");

    obs_properties_add_text(props, "infotext2", "Plugin by <a href=\"https://vrsal.cc/donate\">univrsal</a>, using <a href=\"https://pytorch.org\">libtorch</a> and <a href=\"https://github.com/qiuqiangkong/audioset_tagging_cnn\">PANNs</a>", OBS_TEXT_INFO);
    
    obs_properties_add_bool(props, S_LOG, T_LOG);
    return props;
}

static void mf_defaults(obs_data_t* s)
{
    obs_data_set_default_string(s, S_CLASS_TO_IDENTIFY, "Music");
    obs_data_set_default_double(s, S_CONFIDENCE_THRESHOLD, 0.2);
    obs_data_set_default_double(s, S_SMOOTHING, 0.1);
    obs_data_set_default_int(s, S_WINDOW_SIZE, 1);
}

struct obs_source_info music_filter = {
    .id = "muted_filter",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_AUDIO,
    .get_name = mf_name,
    .create = mf_create,
    .destroy = mf_destroy,
    .get_defaults = mf_defaults,
    .get_properties = mf_properties,
    .update = mf_update,
    .filter_audio = mf_filter_audio,
};

