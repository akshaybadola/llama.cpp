#include "gemma3.h"
#include "gemma3_c.h"

extern "C" {

    static gemma3_context* static_ctx = nullptr;
    static common_sampler* static_sampler = nullptr;
    static common_params* static_params = nullptr;

    void gemma3_print_params(){
        print_common_params(*static_params);
    }

    gemma3_context_t gemma3_static_initialize(const char * model_path, const char * mmproj,
                                              const char * overrides_str) {
        static_params = new common_params(init_params_with_overrides(overrides_str));
        static_params -> model.path = model_path;
        static_params -> mmproj.path = mmproj;
        static_ctx = new gemma3_context(*static_params);
        static_sampler = init_sampler(static_ctx->model, static_params -> sampling);
        return reinterpret_cast<gemma3_context_t>(static_ctx);
    }

    void re_init_sampler(const char * sampler_overrides) {
        common_sampler_free(static_sampler);
        common_params_sampling sampling_params = static_params -> sampling;
        static_sampler = init_sampler(static_ctx->model, static_params -> sampling,
                                      sampler_overrides);
    }

    gemma3_tokens gemma3_tokens_info() {
        return gemma3_tokens {
            static_ctx -> prompt_n,
            static_ctx -> predicted_n
        };
    }

    int gemma3_static_eval_message_text_only(const char * msg_str, bool add_bos) {
        common_chat_msg msg;
        msg.role = "user";
        msg.content = msg_str;
        msg.reasoning_content = "";
        return eval_message_text_only(*static_ctx, msg, add_bos);
    }

    int gemma3_static_eval_message_with_images(const char * msg_or_json_str,
                                               const unsigned char ** images_data,
                                               const int * images_sizes,
                                               const int num_images,
                                               const bool add_bos) {
        std::vector<common_chat_msg> msgs;

        using json = nlohmann::ordered_json;

        try {
            json parsed = nlohmann::json::parse(msg_or_json_str);
            if (parsed.is_array()){
                for (const auto &item : parsed){
                    common_chat_msg msg;
                    for (const auto & [key, val] :item.items()) {
                        if (key == "role"){
                            msg.role = val.get<std::string>();
                        }
                        if (key == "content"){
                            msg.content = val.get<std::string>();
                        }
                    }
                    msgs.push_back(msg);
                }
            } else{
                std::cerr << "JSON parsing error: " << "\nShoud be array of messages\n";
                return 1;
            }

        }  catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Error parsing JSON parameters: " << e.what() << "\n";
            common_chat_msg msg;
            msg.role = "user";
            msg.content = msg_or_json_str;
            msgs.push_back(msg);
        }

        std::vector<ImageData> images(num_images);
        for (int i = 0; i < num_images; ++i) {
            images[i].data = images_data[i];
            images[i].size = images_sizes[i];
        }
        return eval_message_with_images(*static_ctx, msgs, images, add_bos);
    }

    int gemma3_static_generate_response(int n_predict, const char ** stop_strings,
                                        int n_strings) {
        g_is_generating = true;
        int retval = generate_response(*static_ctx, static_sampler, n_predict,
                                       stop_strings, n_strings);
        return retval;
    }

    int gemma3_static_stream_response(token_callback_t py_callback, int n_predict,
                                      const char ** stop_strings,
                                      int n_strings) {
        g_is_generating = true;
        return stream_response(*static_ctx, static_sampler, n_predict, py_callback,
                               stop_strings, n_strings);
    }

    int gemma3_static_collect_response(int n_predict, char* tokens_buffer,
                                       int tokens_buffer_size,
                                       const char ** stop_strings,
                                       int n_strings) {
        g_is_generating = true;
        return collect_response(*static_ctx, static_sampler, n_predict,
                                &tokens_buffer, &tokens_buffer_size,
                                stop_strings, n_strings);
    }

    int gemma3_static_reset() {
        return reset_context(static_ctx);
    }

    void gemma3_static_interrupt() {
        interrupt_generation();
    }

    bool gemma3_is_generating() {
        return g_is_generating;
    }

    common_params_t gemma3_create_params() {
        auto params = new common_params(init_default_params());
        return reinterpret_cast<common_params_t>(params);
    }

    common_params_t gemma3_create_params_with_overrides(const char * json_str) {
        auto * params = new common_params(init_params_with_overrides(json_str));
        return reinterpret_cast<common_params_t>(params);
    }

    common_sampler_t gemma3_create_sampler(gemma3_context_t ctx, common_params_t params) {
        auto* context = reinterpret_cast<gemma3_context*>(ctx);
        auto* cpp_params = reinterpret_cast<common_params*>(params);
        auto* sampler = init_sampler(context->model, cpp_params -> sampling);
        return reinterpret_cast<common_sampler_t>(sampler);
    }

    gemma3_context_t gemma3_create_context(const char * model_path, const char * mmproj,
                                           const char * overrides_str) {
        auto params = new common_params(init_params_with_overrides(overrides_str));
        params -> model.path = model_path;
        params -> mmproj.path = mmproj;
        auto ctx = new gemma3_context(*params);
        return reinterpret_cast<gemma3_context_t>(ctx);
    }

    void gemma3_destroy(gemma3_context_t ctx_ptr) {
        auto ctx = reinterpret_cast<gemma3_context*>(ctx_ptr);
        delete ctx;
    }
}
