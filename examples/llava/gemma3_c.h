#ifdef __cplusplus
extern "C" {
#endif

  typedef void* gemma3_context_t;
  typedef void* gemma3_text_context_t;
  typedef void* common_sampler_t;
  typedef void* common_params_t;
  typedef void (*token_callback_t)(const char* token);
  struct gemma3_tokens {
      int prompt_n;
      int predicted_n;
  };

  // Let there be static
  gemma3_context_t gemma3_static_initialize(const char * model_path, const char * mmproj,
                                            const char * overrides_str);
  void re_init_sampler(const char * sampler_overrides);
  int gemma3_static_eval_message(const char * msg_str, bool add_bos);
  int gemma3_static_eval_message_with_images(const char * msg_str,
                                             const unsigned char ** images_data,
                                             const int * images_sizes,
                                             int num_images,
                                             bool add_bos);
  void gemma3_static_interrupt();
  int gemma3_static_reset();
  void gemma3_print_params();
  int gemma3_static_stream_response(token_callback_t py_callback, int n_predict,
                                    const char ** stop_strings, int n_strings);
  int gemma3_static_generate_response(int n_predict, const char ** stop_strings,int n_strings);
  int gemma3_static_eval_message_text_only(const char * msg_str, bool add_bos);
  int gemma3_static_collect_response(int n_predict, char* tokens_buffer, int tokens_buffer_size,
                                     const char ** stop_strings,int n_strings);
  bool gemma3_is_generating();

  // Create context
  gemma3_context_t gemma3_create_context(const char * model_path, const char * mmproj,
                                         const char * overrides_str);

  // Create Sampler
  common_sampler_t gemma3_create_sampler(gemma3_context_t ctx, common_params_t  params);

  // Create default params
  common_params_t gemma3_create_params(void);

  // Create params with overrides
  common_params_t gemma3_create_params_with_overrides(const char * json_str);

  // Destroy context
  void gemma3_destroy(gemma3_context_t ctx);

  // Evaluate a message (images not supported in C API)
  int gemma3_eval_message(gemma3_context_t ctx_ptr, const char * msg_str);

  // Evaluate a message with images
  int gemma3_eval_message_with_images(gemma3_context_t ctx_ptr, const char * msg_str,
                                      const unsigned char ** images_data,
                                      const int * images_sizes,
                                      int num_images);

  // Generate a response
  int gemma3_generate_response(gemma3_context_t ctx, common_sampler_t smpl, int n_predict);

  // Collect responses in a buffer
  int gemma3_collect_response(gemma3_context_t ctx, common_sampler_t smpl, int n_predict,
                              char* tokens_buffer, int tokens_buffer_size);

  // Stream response with a callback
  int gemma3_stream_response(gemma3_context_t ctx, common_sampler_t smpl, int n_predict,
                             token_callback_t py_callback);


#ifdef __cplusplus
}
#endif
