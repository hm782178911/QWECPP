#include "models/qwen3.cuh"
#include "runtime/device_options.h"
#include "runtime/tokenizer.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
namespace {
struct Message {
    std::string role, content;
};

/**
 * @brief 把多轮消息组织成 Qwen3 非思考聊天输入。
 * @param[in] history 按时间排序的 user/assistant 消息列表，不修改原列表。
 * @return 完整模板文本，末尾带 assistant 前缀及空的 think 块。
 */
std::string prompt(const std::vector<Message> &history) {
    std::string result;
    for (const auto &m : history)
        result += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    return result + "<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

/**
 * @brief 取出可安全显示的完整 UTF-8 字符。
 * @param[in,out] pending 尚未显示的解码字节；已消费的前缀会被移除。
 * @param[in] final true 表示生成结束，不完整尾部按替换字符处理。
 * @return 本次可显示的合法 UTF-8 文本；未完成字符继续留在 pending。
 */
std::string drain(std::string &pending, bool final) {
    std::string out;
    size_t i = 0;
    while (i < pending.size()) {
        unsigned char c = pending[i];
        size_t n = c < 128                  ? 1
                   : c >= 0xc2 && c <= 0xdf ? 2
                   : c >= 0xe0 && c <= 0xef ? 3
                   : c >= 0xf0 && c <= 0xf4 ? 4
                                            : 0;
        if (!n) {
            out += "\xef\xbf\xbd";
            ++i;
            continue;
        }
        if (i + n > pending.size()) {
            if (!final)
                break;
            out += "\xef\xbf\xbd";
            i = pending.size();
            break;
        }
        bool good = true;
        for (size_t j = 1; j < n; ++j) {
            unsigned char b = pending[i + j];
            if ((b & 0xc0) != 0x80)
                good = false;
        }
        if (n >= 3) {
            unsigned char b = pending[i + 1];
            if ((c == 0xe0 && b < 0xa0) || (c == 0xed && b >= 0xa0) || (c == 0xf0 && b < 0x90) ||
                (c == 0xf4 && b >= 0x90))
                good = false;
        }
        if (!good) {
            out += "\xef\xbf\xbd";
            ++i;
            continue;
        }
        out += pending.substr(i, n);
        i += n;
    }
    pending.erase(0, i);
    return out;
}
} // namespace

/**
 * @brief 启动多轮命令行聊天。
 * @param[in] argc 命令行参数数量。
 * @param[in] argv --model DIR [--max-new-tokens N]；stdin 读取用户输入，支持 /clear 和 /exit。
 * @return 正常退出为 0；参数、加载或推理错误为 1。
 */
int main(int argc, char **argv) {
    try {
        qwe::Qwen3Options options;
        std::string directory;
        int max_new = 128;
        bool debug = false;
        bool sliding_window = false;
        using Clock = std::chrono::steady_clock;
        struct Metrics {
            size_t input_tokens;
            int output_tokens;
            double ttft, first_text, total, tpot;
            qwe::TransferStats comm;
            double prefill_ms,decode_ms;
            size_t cache_bytes;
            size_t dropped_turns;
        };
        std::vector<Metrics> metrics;
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b-a).count();
        };
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "Usage: qwecpp_chat --model DIR [--max-new-tokens 128] [--devices "
                             "0,1] [--max-context 2048] [--parallel pp|tp] [--split-layer 14] [--pp-host-transfer] [--kv-cache] [--debug] [--sliding-window]\n";
                return 0;
            }
            if(key=="--kv-cache") { options.kv_cache=true; continue; }
            if (key == "--sliding-window") {
                sliding_window = true;
                continue;
            }
            if (key == "--debug") {
                debug = true;
                options.profile = true;
                continue;
            }
            if (key == "--pp-host-transfer") {
                options.force_host_transfer = true;
                continue;
            }
            if (i + 1 >= argc)
                throw std::runtime_error("missing option value");
            if (key == "--max-context") {
                std::string value=argv[++i]; size_t used;
                options.max_context=std::stoi(value,&used);
                if(used!=value.size() || options.max_context<1 || options.max_context>8192)
                    throw std::runtime_error("max-context must be 1..8192");
            } else if (key == "--parallel") {
                std::string mode = argv[++i];
                if (mode != "tp" && mode != "pp") throw std::runtime_error("parallel must be tp or pp");
                options.tensor_parallel = mode == "tp";
            } else if (key == "--devices")
                options.devices = qwe::parse_devices(argv[++i]);
            else if (key == "--split-layer") {
                std::string value = argv[++i];
                size_t used;
                options.split_layer = std::stoi(value, &used);
                if (used != value.size() || options.split_layer < 1)
                    throw std::runtime_error("split-layer must be positive");
            } else if (key == "--model")
                directory = argv[++i];
            else if (key == "--max-new-tokens") {
                std::string value = argv[++i];
                size_t used;
                max_new = std::stoi(value, &used);
                if (used != value.size() || max_new < 1 || max_new > 256)
                    throw std::runtime_error("max-new-tokens must be 1..256");
            } else
                throw std::runtime_error("unknown option: " + key);
        }
        if (directory.empty())
            throw std::runtime_error("--model required");
        auto load_start = Clock::now();
        qwe::Tokenizer tokenizer(directory);
        if (tokenizer.encode("<|im_start|>") != std::vector<int>{151644} ||
            tokenizer.encode("<|im_end|>") != std::vector<int>{151645})
            throw std::runtime_error("unsupported Qwen3 special token IDs");
        std::cout << "正在加载模型……" << std::endl;
        qwe::Qwen3 model(directory, options);
        std::cout << "可以开始聊天。/clear 清空对话，/exit 退出。" << std::endl;
        double load_ms = ms(load_start, Clock::now());
        std::vector<Message> history;
        std::string line;
        while (std::cout << "你> " << std::flush, std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line == "/exit" || line == "/quit")
                break;
            if (line == "/clear") {
                history.clear();
                std::cout << "已清空对话。\n";
                continue;
            }
            if (line.empty())
                continue;

            model.reset_cache();
            auto request_start = Clock::now();
            // 先在副本中加入输入；超限或编码失败时不污染已有历史。
            auto next = history;
            next.push_back({"user", line});
            std::vector<int> ids;
            size_t dropped_turns = 0;
            try {
                ids = tokenizer.encode(prompt(next));
                if (sliding_window) {
                    // Preserve complete user/assistant pairs and reserve this turn's output.
                    const size_t reserve = std::min<size_t>(max_new, model.context_limit()-1);
                    while (ids.size() + reserve > (size_t)model.context_limit() && next.size() >= 3) {
                        next.erase(next.begin(), next.begin()+2);
                        ++dropped_turns;
                        ids = tokenizer.encode(prompt(next));
                    }
                }
            } catch (const std::exception &e) {
                std::cout << "输入无法处理：" << e.what() << "\n";
                continue;
            }
            const size_t reserve = sliding_window ? std::min<size_t>(max_new, model.context_limit()-1) : 1;
            if (ids.size() + reserve > (size_t)model.context_limit()) {
                std::cout << "上下文超过 " << model.context_limit()
                          << " token 限制，请缩短输入或使用 /clear。\n";
                continue;
            }
            if (sliding_window && dropped_turns)
                std::cout << "[滑动窗口已移除最早的 " << dropped_turns << " 轮对话]\n";
            std::cout << "助手> " << std::flush;
            std::string reply, pending;
            bool stopped = false;
            double prefill_ms=0,decode_ms=0;
            int generated = 0;
            qwe::TransferStats comm;
            size_t input_tokens = ids.size();
            double ttft = -1, first_text = -1;
            Clock::time_point first_token, last_token;
            for (; generated < max_new && ids.size() < (size_t)model.context_limit(); ++generated) {
                // 无 KV Cache：每步重算上下文，只取最后位置的 logits 做贪心选择。
                auto forward_start=Clock::now();
                auto logits = model.forward(ids, true);
                double elapsed=ms(forward_start,Clock::now());
                if(generated==0) prefill_ms+=elapsed; else decode_ms+=elapsed;
                if(debug) {
                    auto step=model.last_transfer_stats();
                    comm.calls+=step.calls; comm.bytes+=step.bytes;
                    comm.p2p_calls+=step.p2p_calls; comm.host_calls+=step.host_calls;
                    comm.total_ms+=step.total_ms; comm.source_wait_ms+=step.source_wait_ms;
                }
                int id = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
                if (id == 151645 || id == 151643) {
                    stopped = true;
                    break;
                }
                auto token_time = Clock::now();
                if (generated == 0) {
                    first_token = token_time;
                    ttft = ms(request_start, token_time);
                }
                last_token = token_time;
                ids.push_back(id);
                pending += tokenizer.decode({id}, true);
                auto text = drain(pending, false);
                reply += text;
                std::cout << text << std::flush;
                if (!text.empty() && first_text < 0)
                    first_text = ms(request_start, Clock::now());
            }
            auto text = drain(pending, true);
            reply += text;
            std::cout << text << "\n";
            if (!text.empty() && first_text < 0)
                first_text = ms(request_start, Clock::now());
            if (debug)
                metrics.push_back({input_tokens, generated, ttft, first_text,
                    ms(request_start, Clock::now()),
                    generated > 1 ? ms(first_token, last_token)/(generated-1) : -1, comm, prefill_ms, decode_ms, model.kv_cache_bytes(), dropped_turns});
            next.push_back({"assistant", reply});
            history = std::move(next);
            if (!stopped) {
                if (ids.size() >= (size_t)model.context_limit())
                    std::cout << "[已达到上下文限制，可用 /clear 开始新对话]\n";
                else
                    std::cout << "[已达到本次回复长度限制]\n";
            }
        }
        std::cout << "再见。\n";
        if (debug) {
            std::cerr << std::fixed << std::setprecision(3)
                      << "[debug] load_ms=" << load_ms << "\n";
            for (size_t i=0; i<metrics.size(); ++i) {
                const auto &m=metrics[i];
                std::cerr << "[debug] turn=" << i+1
                    << " input_tokens=" << m.input_tokens
                    << " dropped_turns=" << m.dropped_turns
                    << " kv_cache=" << options.kv_cache
                    << " kv_cache_bytes=" << m.cache_bytes
                    << " prefill_ms=" << m.prefill_ms << " decode_ms=" << m.decode_ms
                    << " output_tokens=" << m.output_tokens
                    << " ttft_ms=" << m.ttft
                    << " first_text_ms=" << m.first_text
                    << " total_ms=" << m.total
                    << " output_tok_s=" << (m.total>0 ? m.output_tokens*1000.0/m.total : 0)
                    << " tpot_ms=" << m.tpot
                    << " tp_transfer_calls=" << m.comm.calls
                    << " tp_transfer_bytes=" << m.comm.bytes
                    << " tp_p2p_calls=" << m.comm.p2p_calls
                    << " tp_host_calls=" << m.comm.host_calls
                    << " tp_transfer_ms=" << m.comm.total_ms
                    << " tp_source_wait_ms=" << m.comm.source_wait_ms
                    << " tp_transfer_rest_ms=" << m.comm.total_ms-m.comm.source_wait_ms
                    << " tp_transfer_pct=" << (m.total>0 ? m.comm.total_ms*100/m.total : 0)
                    << "\n";
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "chat: " << e.what() << "\n";
        return 1;
    }
}
