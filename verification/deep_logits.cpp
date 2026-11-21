// Compare full logits after an actual teacher-forced 32K or 64K KV history.
// The output format matches verification/summarize.py; only 32 rows are saved.
#include "llama.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 5) {
            throw std::runtime_error("usage: deep_logits MODEL PROMPT OUTPUT HISTORY(32768|65536)");
        }
        const int history = std::stoi(argv[4]);
        if (history != 32768 && history != 65536) throw std::runtime_error("unsupported history");
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("prompt open failed");
        const std::string text((std::istreambuf_iterator<char>(input)), {});
        if (input.bad() || text.size() > INT32_MAX) throw std::runtime_error("prompt read failed or too large");

        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        int count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, false);
        if (count >= 0 || count == INT32_MIN) throw std::runtime_error("tokenization sizing failed");
        std::vector<llama_token> seed(-count);
        count = llama_tokenize(vocab, text.data(), text.size(), seed.data(), seed.size(), true, false);
        if (count < 2) throw std::runtime_error("tokenization failed or prompt too short");

        constexpr int rows = 32;
        constexpr int chunk = 512;
        const int total = history + rows;
        std::vector<llama_token> tokens(total);
        for (int i = 0; i < total; ++i) tokens[i] = seed[i % count];
        auto params = llama_context_default_params();
        params.n_ctx = total + 512;
        params.n_batch = params.n_ubatch = chunk;
        params.n_threads = params.n_threads_batch = 8;
        params.type_k = params.type_v = GGML_TYPE_F16;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.swa_full = false;  // Match the serving path's rolling SWA cache.
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), params), llama_free);
        if (!ctx) throw std::runtime_error("context initialization failed");
        std::unique_ptr<FILE, decltype(&fclose)> out(fopen(argv[3], "wb"), fclose);
        if (!out) throw std::runtime_error("output open failed");
        const int vocab_size = llama_vocab_n_tokens(vocab);
        // decode_tokens=rows records that all emitted rows were decoded singly.
        const int header[] = {0x494c4731, vocab_size, rows, chunk, 99, history, rows, 8};
        if (vocab_size <= 0 || fwrite(header, 4, 8, out.get()) != 8 ||
            fwrite(tokens.data() + history, 4, rows, out.get()) != rows) {
            throw std::runtime_error("header write failed");
        }
        for (int pos = 0; pos < total;) {
            const int batch_size = pos < history ? std::min(chunk, history - pos) : 1;
            auto batch = llama_batch_init(batch_size, 0, 1);
            batch.n_tokens = batch_size;
            for (int i = 0; i < batch_size; ++i) {
                batch.token[i] = tokens[pos + i];
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = pos + i >= history;
            }
            if (llama_decode(ctx.get(), batch) != 0) {
                llama_batch_free(batch);
                throw std::runtime_error("decode failed");
            }
            if (pos >= history) {
                const float * logits = llama_get_logits_ith(ctx.get(), 0);
                if (!logits || fwrite(logits, sizeof(float), vocab_size, out.get()) != size_t(vocab_size)) {
                    llama_batch_free(batch);
                    throw std::runtime_error("missing logits or write failed");
                }
            }
            llama_batch_free(batch);
            pos += batch_size;
        }
        if (fclose(out.release()) != 0) throw std::runtime_error("output close failed");
        ctx.reset();
        model.reset();
        llama_backend_free();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
