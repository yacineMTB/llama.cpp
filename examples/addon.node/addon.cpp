#include "napi.h"
#include "common.h"
#include "llama.h"
#include <chrono>
#include <thread>
#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdint>

static console_state con_st;

static llama_context *g_ctx;
static gpt_params *g_params;

// Initialization function
// Instantiates a global context, and loads the model
// Blocking
Napi::Number init(const Napi::CallbackInfo &info)
{
  Napi::Object obj = info[0].As<Napi::Object>();
  Napi::String modelNapi = obj.Get("model").As<Napi::String>();
  std::string model = modelNapi.Utf8Value();
  Napi::Env env = info.Env();
  llama_init_backend();
  g_params = new gpt_params;
  g_params->model = model;
  // load the model and apply lora adapter, if any
  // TODO: Create a function that holds more than one adapter in memory
  if (obj.Has("lora")) {
    Napi::String loraNapi = obj.Get("lora").As<Napi::String>();
    std::string lora = loraNapi.Utf8Value();
    if (!lora.empty()){
      fprintf(stderr, "Loading lora from Path: %s\n", lora.c_str());
      g_params->lora_adapter = lora;
      g_params->use_mmap = false; // with mmap, ggml lora will throw segfault
    }
  }

  g_ctx = llama_init_from_gpt_params(*g_params);
  fprintf(stderr, "system_info: n_threads = %d / %d | %s\n", g_params->n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
  if (g_ctx == NULL)
  {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return Napi::Number::New(env, 1);
  }

  return Napi::Number::New(env, 0);
}

std::mutex worker_mutex;

// Function to load adapter at runtime
Napi::Number swapLora(const Napi::CallbackInfo &info)
{
  Napi::Object obj = info[0].As<Napi::Object>();
  Napi::String loraNapi = obj.Get("lora").As<Napi::String>();
  std::string lora = loraNapi.Utf8Value();

  fprintf(stderr, "Acquiring lock\n");
  worker_mutex.lock();

  fprintf(stderr, "Swapping lora from Path: %s\n", lora.c_str());
  llama_swap_lora_from_cache(g_ctx, lora.c_str(), get_num_physical_cores(), lora.c_str());

  worker_mutex.unlock();
  return Napi::Number::New(info.Env(), 0);  
}

class InferenceWorker {
public:
  InferenceWorker(Napi::Env env, Napi::Function listener, std::string prompt)
    : listener(Napi::ThreadSafeFunction::New(
          env,
          listener,  // JavaScript function called asynchronously
          "LLaMa Inference Callback",  // Name
          0,  // Unlimited queue
          1,  // Only one thread will use this initially
          [this](Napi::Env) {  // Finalizer used to clean threads up
            nativeThread.join();
          })),
      nativeThread([this, prompt] {
        gpt_params params = *g_params;
        llama_context* ctx = g_ctx;
        params.prompt = prompt;

        std::vector<llama_token> embd_inp;
        // FB tokenizer implementation lurk
        params.prompt.insert(0, 1, ' ');
        embd_inp = ::llama_tokenize(ctx, params.prompt, true);

        const int n_ctx = llama_n_ctx(ctx);
        if ((int)embd_inp.size() > n_ctx - 4)
        {
          fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
          return;
        }

        std::vector<llama_token> last_n_tokens(n_ctx);
        std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
        bool is_antiprompt = false;
        int n_past = 0;
        int n_remain = params.n_predict;
        int n_consumed = 0;

        std::vector<llama_token> embd;

        while ((n_remain != 0 && !is_antiprompt) || params.interactive)
        {
          // predict
          if (embd.size() > 0)
          {
            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int)embd.size(); i += params.n_batch)
            {
              int n_eval = (int)embd.size() - i;
              if (n_eval > params.n_batch)
              {
                n_eval = params.n_batch;
              }
              if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads))
              {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                // Todo handle errors probably
                return;
              }
              n_past += n_eval;
            }
          }
          embd.clear();
          if ((int)embd_inp.size() <= n_consumed)
          {
            // out of user input, sample next token
            const float temp = params.temp;
            const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float top_p = params.top_p;
            const float tfs_z = params.tfs_z;
            const float typical_p = params.typical_p;
            const int32_t repeat_last_n = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float repeat_penalty = params.repeat_penalty;
            const float alpha_presence = params.presence_penalty;
            const float alpha_frequency = params.frequency_penalty;
            const int mirostat = params.mirostat;
            const float mirostat_tau = params.mirostat_tau;
            const float mirostat_eta = params.mirostat_eta;
            const bool penalize_nl = params.penalize_nl;

            llama_token id = 0;
            {
              auto logits = llama_get_logits(ctx);
              auto n_vocab = llama_n_vocab(ctx);

              for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++)
              {
                logits[it->first] += it->second;
              }

              std::vector<llama_token_data> candidates;
              candidates.reserve(n_vocab);
              for (llama_token token_id = 0; token_id < n_vocab; token_id++)
              {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
              }

              llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

              // Apply penalties
              float nl_logit = logits[llama_token_nl()];
              auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
              llama_sample_repetition_penalty(ctx, &candidates_p,
                                              last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                              last_n_repeat, repeat_penalty);
              llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                            last_n_repeat, alpha_frequency, alpha_presence);
              if (!penalize_nl)
              {
                logits[llama_token_nl()] = nl_logit;
              }

              if (temp <= 0)
              {
                // Greedy sampling
                id = llama_sample_token_greedy(ctx, &candidates_p);
              }
              else
              {
                if (mirostat == 1)
                {
                  static float mirostat_mu = 2.0f * mirostat_tau;
                  const int mirostat_m = 100;
                  llama_sample_temperature(ctx, &candidates_p, temp);
                  id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                }
                else if (mirostat == 2)
                {
                  static float mirostat_mu = 2.0f * mirostat_tau;
                  llama_sample_temperature(ctx, &candidates_p, temp);
                  id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                }
                else
                {
                  // Temperature sampling
                  llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                  llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                  llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                  llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                  llama_sample_temperature(ctx, &candidates_p, temp);
                  id = llama_sample_token(ctx, &candidates_p);
                }
              }
              last_n_tokens.erase(last_n_tokens.begin());
              last_n_tokens.push_back(id);
            }
            // add it to the context
            embd.push_back(id);
            // decrement remaining sampling budget
            --n_remain;
          }
          else
          {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp.size() > n_consumed)
            {
              embd.push_back(embd_inp[n_consumed]);
              last_n_tokens.erase(last_n_tokens.begin());
              last_n_tokens.push_back(embd_inp[n_consumed]);
              ++n_consumed;
              if ((int)embd.size() >= params.n_batch)
              {
                break;
              }
            }
          }

          for (auto id : embd)
          {
            const char* tokenStr = llama_token_to_str(ctx, id);
            this->listener.NonBlockingCall([tokenStr](Napi::Env env, Napi::Function jsCallback) {
              jsCallback.Call({ Napi::String::New(env, tokenStr) });
            });
          }
          // end of text token
          if (!embd.empty() && embd.back() == llama_token_eos())
          {
            this->listener.NonBlockingCall([](Napi::Env env, Napi::Function jsCallback) {
              jsCallback.Call({ Napi::String::New(env, "[end of text]") });
            });
            break;
          }
        }
        worker_mutex.unlock();
        this->listener.Release();
      }) {}
private:
  Napi::ThreadSafeFunction listener;
  std::thread nativeThread;
};

InferenceWorker* global_worker;
Napi::Value StartAsync(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Object options = info[0].As<Napi::Object>();
  Napi::Function listener = options.Get("listener").As<Napi::Function>();
  std::string prompt = options.Get("prompt").ToString();

  // Already a worker running
  if (!worker_mutex.try_lock()) {
    return Napi::Boolean::New(env, false);
  }

  delete global_worker;
  global_worker = nullptr;
  global_worker = new InferenceWorker(env, listener, prompt);

  return Napi::Boolean::New(env, true);
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
  exports.Set(Napi::String::New(env, "init"),
              Napi::Function::New(env, init));
  exports.Set(
      Napi::String::New(env, "startAsync"),
      Napi::Function::New(env, StartAsync));
  exports.Set(
      Napi::String::New(env, "swapLora"),
      Napi::Function::New(env, swapLora));
  return exports;
}

NODE_API_MODULE(addon, Init);
