#include "log.h"
#include "chat.h"
#include "mtmd.h"
#include "sampling.h"
#include "arg.h"
#include "json.hpp"
#include "llama.h"

#include <array>
#include <deque>
#include <iostream>
#include <cstring>
#include <vector>


using json = nlohmann::ordered_json;


static void print_common_params(const common_params& p) {
    using std::cout;
    using std::endl;

    #define PRINT_FIELD(x) cout << #x << ": " << p.x << "\n"
    #define PRINT_SAMPLING_FIELD(x) cout << "\t" << #x << ": " << p.sampling.x << "\n"

    PRINT_FIELD(n_predict);
    PRINT_FIELD(n_ctx);
    PRINT_FIELD(n_batch);
    PRINT_FIELD(n_ubatch);
    PRINT_FIELD(n_keep);
    PRINT_FIELD(n_chunks);
    PRINT_FIELD(n_parallel);
    PRINT_FIELD(n_sequences);
    PRINT_FIELD(grp_attn_n);
    PRINT_FIELD(grp_attn_w);
    PRINT_FIELD(n_print);
    PRINT_FIELD(rope_freq_base);
    PRINT_FIELD(rope_freq_scale);
    PRINT_FIELD(yarn_ext_factor);
    PRINT_FIELD(yarn_attn_factor);
    PRINT_FIELD(yarn_beta_fast);
    PRINT_FIELD(yarn_beta_slow);
    PRINT_FIELD(yarn_orig_ctx);
    PRINT_FIELD(defrag_thold);

    cout << "devices: " << p.devices.size() << " entries" << "\n";

    PRINT_FIELD(n_gpu_layers);
    PRINT_FIELD(main_gpu);
    for (int i = 0; i < 128; ++i) {
        if (p.tensor_split[i] != 0.0f)
            {
                cout << "tensor_split[" << i << "]: " << p.tensor_split[i] << "\n";
            }
    }

    PRINT_FIELD(split_mode);
    // You will need to define print_cpu_params(), etc., if you want more than address dumps:
    cout << "cpuparams: " << &p.cpuparams << "\n";
    cout << "cpuparams_batch: " << &p.cpuparams_batch << "\n";
    cout << "cb_eval: " << (void*)p.cb_eval << "\n";
    cout << "cb_eval_user_data: " << p.cb_eval_user_data << "\n";
    PRINT_FIELD(numa);
    PRINT_FIELD(rope_scaling_type);
    PRINT_FIELD(pooling_type);
    PRINT_FIELD(attention_type);

    cout << "sampling:\n";

    PRINT_SAMPLING_FIELD(n_prev);
    PRINT_SAMPLING_FIELD(n_probs);
    PRINT_SAMPLING_FIELD(min_keep);
    PRINT_SAMPLING_FIELD(top_k);
    PRINT_SAMPLING_FIELD(top_p);
    PRINT_SAMPLING_FIELD(min_p);
    PRINT_SAMPLING_FIELD(xtc_probability);
    PRINT_SAMPLING_FIELD(xtc_threshold);
    PRINT_SAMPLING_FIELD(typ_p);
    PRINT_SAMPLING_FIELD(temp);
    PRINT_SAMPLING_FIELD(dynatemp_range);
    PRINT_SAMPLING_FIELD(dynatemp_exponent);
    PRINT_SAMPLING_FIELD(penalty_last_n);
    PRINT_SAMPLING_FIELD(penalty_repeat);
    PRINT_SAMPLING_FIELD(penalty_freq);
    PRINT_SAMPLING_FIELD(penalty_present);
    PRINT_SAMPLING_FIELD(dry_multiplier);
    PRINT_SAMPLING_FIELD(dry_base);
    PRINT_SAMPLING_FIELD(dry_allowed_length);
    PRINT_SAMPLING_FIELD(dry_penalty_last_n);
    PRINT_SAMPLING_FIELD(mirostat);
    PRINT_SAMPLING_FIELD(top_n_sigma);
    PRINT_SAMPLING_FIELD(mirostat_tau);
    PRINT_SAMPLING_FIELD(mirostat_eta);
    PRINT_SAMPLING_FIELD(ignore_eos);
    PRINT_SAMPLING_FIELD(no_perf);
    PRINT_SAMPLING_FIELD(timing_per_token);
    PRINT_SAMPLING_FIELD(grammar);

    cout << "speculative: " << &p.speculative << "\n";
    cout << "vocoder: " << &p.vocoder << "\n";
    cout << "model: " << &p.model << "\n";

    PRINT_FIELD(model_alias);
    PRINT_FIELD(hf_token);
    PRINT_FIELD(prompt);
    PRINT_FIELD(system_prompt);
    PRINT_FIELD(prompt_file);
    PRINT_FIELD(path_prompt_cache);
    PRINT_FIELD(input_prefix);
    PRINT_FIELD(input_suffix);
    PRINT_FIELD(lookup_cache_static);
    PRINT_FIELD(lookup_cache_dynamic);
    PRINT_FIELD(logits_file);

    PRINT_FIELD(in_files.size());
    PRINT_FIELD(antiprompt.size());
    PRINT_FIELD(kv_overrides.size());
    PRINT_FIELD(tensor_buft_overrides.size());
    PRINT_FIELD(lora_init_without_apply);
    PRINT_FIELD(lora_adapters.size());
    PRINT_FIELD(control_vectors.size());

    PRINT_FIELD(verbosity);
    PRINT_FIELD(control_vector_layer_start);
    PRINT_FIELD(control_vector_layer_end);
    PRINT_FIELD(ppl_stride);
    PRINT_FIELD(ppl_output_type);
    PRINT_FIELD(hellaswag);
    PRINT_FIELD(hellaswag_tasks);
    PRINT_FIELD(winogrande);
    PRINT_FIELD(winogrande_tasks);
    PRINT_FIELD(multiple_choice);
    PRINT_FIELD(multiple_choice_tasks);
    PRINT_FIELD(kl_divergence);
    PRINT_FIELD(usage);
    PRINT_FIELD(completion);
    PRINT_FIELD(use_color);
    PRINT_FIELD(special);
    PRINT_FIELD(interactive);
    PRINT_FIELD(interactive_first);
    PRINT_FIELD(prompt_cache_all);
    PRINT_FIELD(prompt_cache_ro);
    PRINT_FIELD(escape);
    PRINT_FIELD(multiline_input);
    PRINT_FIELD(simple_io);
    PRINT_FIELD(cont_batching);
    PRINT_FIELD(flash_attn);
    PRINT_FIELD(no_perf);
    PRINT_FIELD(ctx_shift);
    PRINT_FIELD(input_prefix_bos);
    PRINT_FIELD(logits_all);
    PRINT_FIELD(use_mmap);
    PRINT_FIELD(use_mlock);
    PRINT_FIELD(verbose_prompt);
    PRINT_FIELD(display_prompt);
    PRINT_FIELD(dump_kv_cache);
    PRINT_FIELD(no_kv_offload);
    PRINT_FIELD(warmup);
    PRINT_FIELD(check_tensors);
    PRINT_FIELD(single_turn);
    PRINT_FIELD(cache_type_k);
    PRINT_FIELD(cache_type_v);
    PRINT_FIELD(conversation_mode);

    cout << "mmproj: " << &p.mmproj << "\n";
    PRINT_FIELD(image.size());
    PRINT_FIELD(embedding);
    PRINT_FIELD(embd_normalize);
    PRINT_FIELD(embd_out);
    PRINT_FIELD(embd_sep);
    PRINT_FIELD(reranking);

    PRINT_FIELD(port);
    PRINT_FIELD(timeout_read);
    PRINT_FIELD(timeout_write);
    PRINT_FIELD(n_threads_http);
    PRINT_FIELD(n_cache_reuse);
    PRINT_FIELD(hostname);
    PRINT_FIELD(public_path);
    PRINT_FIELD(chat_template);
    PRINT_FIELD(use_jinja);
    PRINT_FIELD(enable_chat_template);
    PRINT_FIELD(reasoning_format);
    PRINT_FIELD(api_keys.size());
    PRINT_FIELD(ssl_file_key);
    PRINT_FIELD(ssl_file_cert);
    PRINT_FIELD(webui);
    PRINT_FIELD(endpoint_slots);
    PRINT_FIELD(endpoint_props);
    PRINT_FIELD(endpoint_metrics);
    PRINT_FIELD(log_json);
    PRINT_FIELD(slot_save_path);
    PRINT_FIELD(slot_prompt_similarity);
    PRINT_FIELD(is_pp_shared);
    PRINT_FIELD(n_pp.size());
    PRINT_FIELD(n_tg.size());
    PRINT_FIELD(n_pl.size());
    PRINT_FIELD(context_files.size());
    PRINT_FIELD(chunk_size);
    PRINT_FIELD(chunk_separator);
    PRINT_FIELD(n_junk);
    PRINT_FIELD(i_pos);
    PRINT_FIELD(n_out_freq);
    PRINT_FIELD(n_save_freq);
    PRINT_FIELD(i_chunk);
    PRINT_FIELD(process_output);
    PRINT_FIELD(compute_ppl);
    PRINT_FIELD(n_pca_batch);
    PRINT_FIELD(n_pca_iterations);
    PRINT_FIELD(cvector_dimre_method);
    PRINT_FIELD(cvector_positive_file);
    PRINT_FIELD(cvector_negative_file);
    PRINT_FIELD(spm_infill);
    PRINT_FIELD(batched_bench_output_jsonl);
    PRINT_FIELD(out_file);

    #undef PRINT_FIELD
}


struct ImageData {
    const unsigned char* data;
    size_t size;
};


struct gemma3_context {
    mtmd_context_ptr ctx_vision = nullptr;
    common_init_result llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    llama_batch         batch;
    int                 n_batch;

    // note: we know that gemma3 template is "linear", meaning each turn is completely separated to another
    // so here we don't need to keep track of chat history
    common_chat_templates_ptr tmpls;

    int n_threads    = 1;
    llama_pos n_past = 0;

    gemma3_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        vocab = llama_model_get_vocab(model);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(params.n_batch, 0, 1);
        n_batch = params.n_batch;
        tmpls = common_chat_templates_init(model, params.chat_template);
        init_vision_context(params);
    }

    void init_vision_context(common_params & params) {
        if (params.mmproj.path.empty()){
            return;
        }
        const char * clip_path = params.mmproj.path.c_str();
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mtmd_context_params{
            /* use_gpu */   true,
            /* timings */   true,
            /* n_threads */ params.cpuparams.n_threads,
            /* verbosity */ GGML_LOG_LEVEL_INFO,
        }));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }
};


static bool match_reversed(std::deque<std::string>text_vecs, std::vector<std::string> patterns){
    std::string text;
    for (size_t i = 0; i < text_vecs.size(); i++){
        text += text_vecs[i];
        if (i == 128) {
            break;
        }
    }
    for (auto& pattern : patterns ){
        bool match = true;
        for (size_t i = 0; i < pattern.size(); ++i) {
            if (text[i] != pattern[i]) {
                match = false;
                // std::string sub = text.substr(0, i);
                // std::cout << "No match at: " << sub << "\n";
                break;
            }
            if (i > 128){
                match = false;
                // std::string sub = text.substr(0, i);
                // std::cout << "No match at: " << sub << "\n";
                break;
            }
        }
        if (match) {
            // std::cout << "Got match: " << pattern << "\n";
            return true; // Found a match
        }
    }
    return false; // No match found
}


static bool g_is_generating;

static common_params init_default_params() {
    common_params params;
    params.sampling.temp = 0.2; // lower temp by default for better quality

    char * argv[] = {(char *)"test"};

    if (!common_params_parse(1, argv, params, LLAMA_EXAMPLE_LLAVA)) {
        return params;
    }

    /* print_common_params(params); */
    return params;
}


static common_params init_params_with_overrides(const char* json_str) {
    common_params params = init_default_params(); // Start with defaults

    if (json_str != nullptr && std::strlen(json_str) > 0) {
        try {
            json args_map = nlohmann::json::parse(json_str);

            // Iterate through the JSON object and update params
            for (auto const& [key, val] : args_map.items()) {
                if (key == "n_ctx" && val.is_number_integer()) {
                    params.n_ctx = val.get<int32_t>();
                    printf("Changed n_ctx\n");
                } else if (key == "n_batch" && val.is_number_integer()) {
                    params.n_batch = val.get<int32_t>();
                    printf("Changed n_batch\n");
                } else if (key == "n_chunks" && val.is_number_integer()) {
                    params.n_chunks = val.get<int32_t>();
                    printf("Changed n_chunks\n");
                } else if (key == "n_print" && val.is_number_integer()) {
                    params.n_print = val.get<int32_t>();
                    printf("Changed n_print\n");
                } else if (key == "flash_attn" && val.is_boolean()) {
                    params.flash_attn = val.get<bool>();
                    printf("Changed flash_attn\n");
                } else if (key == "n_gpu_layers" && val.is_number_integer()) {
                    params.n_gpu_layers = val.get<int32_t>();
                    printf("Changed n_gpu_layers\n");
                } else if (key == "main_gpu" && val.is_number_integer()) {
                    params.main_gpu = val.get<int32_t>();
                    printf("Changed main_gpu\n");
                } else if (key == "tensor_split" && val.is_array()) {
                    size_t num_elements = 0;
                    for (const auto& element : val) {
                        params.tensor_split[num_elements++] = element.get<float>();
                    }
                } else if (key == "log_file" && val.is_string()) {
                    common_log_set_file(common_log_main(), val.get<std::string>().c_str());
                    printf("Set log_file to %s\n", val.get<std::string>().c_str());
                } else {
                    std::cerr << "Warning: Unknown, unsupported or invalid parameter: " << key << "\n";
                }
            }
            /* print_common_params(params); // Print updated params */
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Error parsing JSON parameters: " << e.what() << "\n";
        }
    }
    fflush(stdout);
    return params;
}

static common_sampler * init_sampler(llama_model * model,
                                     common_params_sampling & sampling_params,
                                     const char * overrides_str = ""){
    if (overrides_str != nullptr && std::strlen(overrides_str) > 0) {
        try {
            json args_map = nlohmann::json::parse(overrides_str);
            for (auto const& [key, val] : args_map.items()) {

                // Integer params
                if (key == "seed" && val.is_number_integer()) {
                    sampling_params.seed = val.get<uint32_t>();
                    printf("Changed sampling params.seed to %u\n", sampling_params.seed);
                } else if (key == "n_prev" && val.is_number_integer()) {
                    sampling_params.n_prev = val.get<int32_t>();
                    printf("Changed sampling params.n_prev to %d\n", sampling_params.n_prev);
                } else if (key == "n_probs" && val.is_number_integer()) {
                    sampling_params.n_probs = val.get<int32_t>();
                    printf("Changed sampling params.n_probs to %d\n", sampling_params.n_probs);
                } else if (key == "min_keep" && val.is_number_integer()) {
                    sampling_params.min_keep = val.get<int32_t>();
                    printf("Changed sampling params.min_keep to %d\n", sampling_params.min_keep);
                } else if (key == "top_k" && val.is_number_integer()) {
                    sampling_params.top_k = val.get<int32_t>();
                    printf("Changed sampling params.top_k to %d\n", sampling_params.top_k);
                } else if (key == "penalty_last_n" && val.is_number_integer()) {
                    sampling_params.penalty_last_n = val.get<int32_t>();
                    printf("Changed sampling params.penalty_last_n to %d\n", sampling_params.penalty_last_n);
                } else if (key == "dry_allowed_length" && val.is_number_integer()) {
                    sampling_params.dry_allowed_length = val.get<int32_t>();
                    printf("Changed sampling params.dry_allowed_length to %d\n", sampling_params.dry_allowed_length);
                } else if (key == "dry_penalty_last_n" && val.is_number_integer()) {
                    sampling_params.dry_penalty_last_n = val.get<int32_t>();
                    printf("Changed sampling params.dry_penalty_last_n to %d\n", sampling_params.dry_penalty_last_n);
                } else if (key == "mirostat" && val.is_number_integer()) {
                    sampling_params.mirostat = val.get<int32_t>();
                    printf("Changed sampling params.mirostat to %d\n", sampling_params.mirostat);

                    // Float params
                } else if (key == "top_p" && val.is_number_float()) {
                    sampling_params.top_p = val.get<float>();
                    printf("Changed sampling params.top_p to %f\n", sampling_params.top_p);
                } else if (key == "min_p" && val.is_number_float()) {
                    sampling_params.min_p = val.get<float>();
                    printf("Changed sampling params.min_p to %f\n", sampling_params.min_p);
                } else if (key == "xtc_probability" && val.is_number_float()) {
                    sampling_params.xtc_probability = val.get<float>();
                    printf("Changed sampling params.xtc_probability to %f\n", sampling_params.xtc_probability);
                } else if (key == "xtc_threshold" && val.is_number_float()) {
                    sampling_params.xtc_threshold = val.get<float>();
                    printf("Changed sampling params.xtc_threshold to %f\n", sampling_params.xtc_threshold);
                } else if (key == "typ_p" && val.is_number_float()) {
                    sampling_params.typ_p = val.get<float>();
                    printf("Changed sampling params.typ_p to %f\n", sampling_params.typ_p);
                } else if (key == "temp" && val.is_number_float()) {
                    sampling_params.temp = val.get<float>();
                    printf("Changed sampling params.temp to %f\n", sampling_params.temp);
                } else if (key == "dynatemp_range" && val.is_number_float()) {
                    sampling_params.dynatemp_range = val.get<float>();
                    printf("Changed sampling params.dynatemp_range to %f\n", sampling_params.dynatemp_range);
                } else if (key == "dynatemp_exponent" && val.is_number_float()) {
                    sampling_params.dynatemp_exponent = val.get<float>();
                    printf("Changed sampling params.dynatemp_exponent to %f\n", sampling_params.dynatemp_exponent);
                } else if (key == "penalty_repeat" && val.is_number_float()) {
                    sampling_params.penalty_repeat = val.get<float>();
                    printf("Changed sampling params.penalty_repeat to %f\n", sampling_params.penalty_repeat);
                } else if (key == "penalty_freq" && val.is_number_float()) {
                    sampling_params.penalty_freq = val.get<float>();
                    printf("Changed sampling params.penalty_freq to %f\n", sampling_params.penalty_freq);
                } else if (key == "penalty_present" && val.is_number_float()) {
                    sampling_params.penalty_present = val.get<float>();
                    printf("Changed sampling params.penalty_present to %f\n", sampling_params.penalty_present);
                } else if (key == "dry_multiplier" && val.is_number_float()) {
                    sampling_params.dry_multiplier = val.get<float>();
                    printf("Changed sampling params.dry_multiplier to %f\n", sampling_params.dry_multiplier);
                } else if (key == "dry_base" && val.is_number_float()) {
                    sampling_params.dry_base = val.get<float>();
                    printf("Changed sampling params.dry_base to %f\n", sampling_params.dry_base);
                } else if (key == "top_n_sigma" && val.is_number_float()) {
                    sampling_params.top_n_sigma = val.get<float>();
                    printf("Changed sampling params.top_n_sigma to %f\n", sampling_params.top_n_sigma);
                } else if (key == "mirostat_tau" && val.is_number_float()) {
                    sampling_params.mirostat_tau = val.get<float>();
                    printf("Changed sampling params.mirostat_tau to %f\n", sampling_params.mirostat_tau);
                } else if (key == "mirostat_eta" && val.is_number_float()) {
                    sampling_params.mirostat_eta = val.get<float>();
                    printf("Changed sampling params.mirostat_eta to %f\n", sampling_params.mirostat_eta);

                    // Boolean params
                } else if (key == "ignore_eos" && val.is_boolean()) {
                    sampling_params.ignore_eos = val.get<bool>();
                    printf("Changed sampling params.ignore_eos to %s\n",
                           sampling_params.ignore_eos ? "true" : "false");
                } else if (key == "no_perf" && val.is_boolean()) {
                    sampling_params.no_perf = val.get<bool>();
                    printf("Changed sampling params.no_perf to %s\n", sampling_params.no_perf ? "true" : "false");
                } else if (key == "timing_per_token" && val.is_boolean()) {
                    sampling_params.timing_per_token = val.get<bool>();
                    printf("Changed sampling params.timing_per_token to %s\n",
                           sampling_params.timing_per_token ? "true" : "false");

                } else if (key == "grammar" && val.is_string()) {
                    sampling_params.grammar = val.get<std::string>();
                    printf("Changed sampling params.grammar to %s\n", sampling_params.grammar.c_str());
                } else if (key == "grammar_lazy" && val.is_boolean()) {
                    sampling_params.grammar_lazy = val.get<bool>();
                    printf("Changed sampling params.grammar_lazy to %s\n",
                           sampling_params.grammar_lazy? "true": "false");
                // TODO: Grammar triggers are too complicated
                // } else if (key == "grammar_triggers" && val.is_array()) {
                //     sampling_params.preserved_tokens = val.get<std::array<std::string, size_t Nm>>();
                //     printf("Changed sampling params.grammar to %s\n", sampling_params.grammar.c_str());

                    // Unknown or unsupported parameter
                } else {
                    std::cerr << "Warning: Unknown, unsupported or invalid parameter: " << key << "\n";
                }
            }
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Error parsing JSON parameters: " << e.what() << "\n"
                      << "Making default sampler" << "\n";
            common_sampler * sampler = common_sampler_init(model, sampling_params);
            return sampler;
        }
    }
    fflush(stdout);
    common_sampler * sampler = common_sampler_init(model, sampling_params);
    return sampler;
}


typedef void (*token_callback_t)(const char* token);

static int stream_response(gemma3_context & ctx, common_sampler * sampler, int n_predict,
                           token_callback_t py_callback, const char ** stop_tokens_ptr,
                           int num_strings) {
    std::vector<std::string> stop_strings;
    for (int i = 0; i < num_strings; ++i) {
        std::string text = stop_tokens_ptr[i];
        std::reverse(text.begin(), text.end());
        stop_strings.push_back(text);
    }
    std::deque<std::string> stop;
    std::cout << "\nStreaming response\nN Past: " << ctx.n_past
              << ", Generating\n";
    fflush(stdout);
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict) {
            printf("Reached n_predict\n");
            break;
        }
        if (!g_is_generating) {
            printf("Generation flag is unset\n");
            LOG_INF("Generated %d tokens\n", i);
            break;
        }

        llama_token token_id = common_sampler_sample(sampler, ctx.lctx, -1);
        common_sampler_accept(sampler, token_id, true);


        if (llama_vocab_is_eog(ctx.vocab, token_id)) {
            // printf("\nGOT EOS\n");
            // fflush(stdout);

            LOG_INF("Generated %d tokens\n", i);
            g_is_generating = false;
            py_callback("[EOS]");
            break; // end of generation
        }


        std::string token_str = common_token_to_piece(ctx.lctx, token_id);

        std::string reversed = token_str;
        std::reverse(reversed.begin(), reversed.end());
        stop.push_front(reversed);

        if (match_reversed(stop, stop_strings)){
            py_callback(token_str.c_str());
            LOG_INF("Generated %d tokens\n", i);
            g_is_generating = false;
            py_callback("[EOS]");
            break; // end of generation
        }

        // printf("GOT TOKEN %s\n", token_str.c_str());
        // fflush(stdout);

        // Call the Python callback function with the generated token
        if (py_callback != nullptr) {
            py_callback(token_str.c_str());
        }

        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            LOG_INF("Generated %d tokens\n", i);
            g_is_generating = false;
            py_callback("[EOS]");
            return 1;
        }
    }
    g_is_generating = false;
    py_callback("[EOS]");
    return 0;
}


static int generate_response(gemma3_context & ctx, common_sampler * sampler, int n_predict,
                             const char ** stop_tokens_ptr,
                             int num_strings) {
    std::vector<std::string> stop_strings;
    for (int i = 0; i < num_strings; ++i) {
        std::string text = stop_tokens_ptr[i];
        std::reverse(text.begin(), text.end());
        stop_strings.push_back(text);
    }
    std::deque<std::string> stop;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating) {
            printf("\n");
            break;
        }

        if (!g_is_generating) {
            printf("Generation flag is unset\n");
            LOG_INF("Generated %d tokens\n", i);
            break;
        }

        llama_token token_id = common_sampler_sample(sampler, ctx.lctx, -1);
        common_sampler_accept(sampler, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id)) {
            printf("\n");
            LOG_INF("Generated %d tokens\n", i);
            g_is_generating = false;
            break; // end of generation
        }

        std::string token_str = common_token_to_piece(ctx.lctx, token_id);
        std::string reversed = token_str;
        std::reverse(reversed.begin(), reversed.end());
        stop.push_front(reversed);

        if (match_reversed(stop, stop_strings)){
            printf("%s", token_str.c_str());
            printf("\n");
            LOG_INF("Generated %d tokens\n", i);
            g_is_generating = false;
            break; // end of generation
        }

        printf("%s", token_str.c_str());
        fflush(stdout);

        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }
    return 0;
}


static int collect_response(gemma3_context & ctx, common_sampler * sampler, int n_predict,
                            char** tokens_buffer, const int* tokens_buffer_size,
                            const char ** stop_tokens_ptr, int num_strings) {
    std::vector<std::string> stop_strings;
    for (int i = 0; i < num_strings; ++i) {
        std::string text = stop_tokens_ptr[i];
        std::reverse(text.begin(), text.end());
        stop_strings.push_back(text);
    }
    std::deque<std::string> stop;
    std::vector<std::string> generated_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating) {
            printf("\n");
            break;
        }

        if (!g_is_generating) {
            printf("Generation flag is unset\n");
            LOG_INF("Generated %d tokens\n", i);
            break;
        }

        llama_token token_id = common_sampler_sample(sampler, ctx.lctx, -1);
        common_sampler_accept(sampler, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id)) {
            printf("\n");
            break; // end of generation
        }

        std::string sampled_token = common_token_to_piece(ctx.lctx, token_id);

        std::string reversed = sampled_token;
        std::reverse(reversed.begin(), reversed.end());
        stop.push_front(reversed);

        if (match_reversed(stop, stop_strings)){
            generated_tokens.push_back(sampled_token);
            LOG_INF("Generated %d tokens\n", i);
            g_is_generating = false;
            break; // end of generation
        }

        printf("%s", sampled_token.c_str());
        fflush(stdout);

        generated_tokens.push_back(sampled_token);

        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }
    if (!generated_tokens.empty()) {
      size_t total_size = 0;
      for (const auto& token : generated_tokens) {
        total_size += token.length() + 1; // +1 for null terminator
      }

      if (*tokens_buffer_size < total_size) {
        // Buffer is too small, inform the caller
        return -2; // Indicate buffer too small
      }

      // Copy tokens into the buffer
      char* current_pos = *tokens_buffer;
      for (const auto& token : generated_tokens) {
        strcpy(current_pos, token.c_str());
        current_pos += token.length();
      }
      LOG_INF("Generated %zu tokens\n", generated_tokens.size());
      return generated_tokens.size();
    }
    return 0;
}


static int eval_message_with_images(gemma3_context & ctx,
                                    common_chat_msg & msg,
                                    std::vector<ImageData> & images,
                                    bool add_bos = false) {
    std::vector<mtmd_bitmap> bitmaps;

    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false; // jinja is buggy here
    auto formatted_chat = common_chat_templates_apply(ctx.tmpls.get(), tmpl_inputs);

    for (auto & image_buf : images) {
        mtmd_bitmap bitmap;
        if (mtmd_helper_bitmap_init_from_buf(image_buf.data, image_buf.size, bitmap)) {
            LOG_ERR("Unable to load image\n");
            return 2; // image not found
        }
        bitmaps.push_back(std::move(bitmap));
    }

    mtmd_input_text text;
    text.text          = formatted_chat.prompt;
    text.add_special   = add_bos;
    text.parse_special = true;
    LOG_INF("Formatted_chat.prompt: %s\n\n", formatted_chat.prompt.c_str());
    mtmd_input_chunks chunks;
    int32_t           res = mtmd_tokenize(ctx.ctx_vision.get(), chunks, text, bitmaps);
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }
    if (mtmd_helper_eval(ctx.ctx_vision.get(), ctx.lctx, chunks, ctx.n_past, 0, ctx.n_batch)) {
        LOG_ERR("Unable to eval prompt\n");
        return 1;
    }
    ctx.n_past += mtmd_helper_get_n_tokens(chunks);
    LOG_INF("Got %d tokens\n\n", ctx.n_past);
    return 0;
}

static int eval_message_text_only(gemma3_context & ctx,
                                  common_chat_msg & msg,
                                  bool add_bos = false) {
    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false; // jinja is buggy here
    auto formatted_chat = common_chat_templates_apply(ctx.tmpls.get(), tmpl_inputs);

    LOG_INF("Eval text only\nformatted_chat.prompt: %s\n", formatted_chat.prompt.c_str());

    const auto * vocab = llama_model_get_vocab(ctx.model);
    auto prompt = formatted_chat.prompt;
    const int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, add_bos, true);
    std::vector<llama_token> prompt_tokens(n_tokens);

    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(),
                       add_bos, true) < 0) {
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    // int n_tokens = formatted_chat.prompt.length() + 2;
    // std::vector<llama_token> result(n_tokens);

    // printf("\nHERE\n\n");
    // fflush(stdout);

    // if (n_tokens < 0) {
    //     result.resize(-n_tokens);
    //     int check = llama_tokenize(vocab, formatted_chat.prompt.data(), formatted_chat.prompt.length(),
    //                                result.data(), formatted_chat.prompt.size(),
    //                                add_bos, true);
    //     GGML_ASSERT(check == -n_tokens);
    // } else {
    //     result.resize(n_tokens);
    // }

    ctx.n_past += n_tokens;

    return 0;
}


static int eval_message(gemma3_context & ctx, common_chat_msg & msg,
                        std::vector<std::string> & images_fname,
                        bool add_bos = false) {
    std::vector<mtmd_bitmap> bitmaps;

    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false; // jinja is buggy here
    auto formatted_chat = common_chat_templates_apply(ctx.tmpls.get(), tmpl_inputs);
    LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.prompt.c_str());

    for (auto & fname : images_fname) {
        mtmd_bitmap bitmap;
        if (mtmd_helper_bitmap_init_from_file(fname.c_str(), bitmap)) {
            LOG_ERR("Unable to load image %s\n", fname.c_str());
            return 2; // image not found
        }
        bitmaps.push_back(std::move(bitmap));
    }

    mtmd_input_text text;
    text.text          = formatted_chat.prompt;
    text.add_special   = add_bos;
    text.parse_special = true;
    mtmd_input_chunks chunks;
    int32_t           res = mtmd_tokenize(ctx.ctx_vision.get(), chunks, text, bitmaps);
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }
    if (mtmd_helper_eval(ctx.ctx_vision.get(), ctx.lctx, chunks, ctx.n_past, 0, ctx.n_batch)) {
        LOG_ERR("Unable to eval prompt\n");
        return 1;
    }
    ctx.n_past += mtmd_helper_get_n_tokens(chunks);
    return 0;
}


static int reset_context(gemma3_context * ctx){
    try{
        ctx -> n_past = 0;
        // TODO: What does this seq_id do?
        llama_kv_self_seq_rm(ctx -> lctx, 0, 0, -1); // DON'T keep BOS
        return 0;
    }  catch (const std::exception & e) {
        LOG_INF("%s: Could not reset context: %s\n", __func__,  e.what());
        return 1;
    }
}

static void interrupt_generation(){
    g_is_generating = false;
}
