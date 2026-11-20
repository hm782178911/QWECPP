#include "runtime/text_model.cuh"
#include "runtime/tokenizer.h"
#include "runtime/device_options.h"
#include "backend/cuda/device_guard.cuh"
#include "httplib.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <filesystem>
using Clock = std::chrono::steady_clock;
using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
static std::string dump(cJSON *j) {
    char *p = cJSON_PrintUnformatted(j);
    if (!p)
        throw std::bad_alloc();
    std::string s = p;
    cJSON_free(p);
    return s;
}
static Json parse(const std::string &s) {
    return Json(cJSON_Parse(s.c_str()), cJSON_Delete);
}
static void error(httplib::Response &res, int code, const std::string &message) {
    Json j(cJSON_CreateObject(), cJSON_Delete);
    cJSON_AddStringToObject(j.get(), "error", message.c_str());
    res.status = code;
    res.set_content(dump(j.get()), "application/json; charset=utf-8");
}
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
// 保留不完整UTF-8尾部，避免token边界把一个字符拆成多条非法JSON文本。
static std::string drain(std::string &s) {
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        size_t n = c < 128 ? 1 : c < 224 ? 2 : c < 240 ? 3 : 4;
        if (i + n > s.size())
            break;
        i += n;
    }
    auto text = s.substr(0, i);
    s.erase(0, i);
    return text;
}
// 只接受内联data URI，不访问远程URL或本地文件路径，防止请求触发任意资源读取。
static std::vector<uint8_t> decode_image_uri(const std::string &uri) {
    const std::string png = "data:image/png;base64,", jpeg = "data:image/jpeg;base64,";
    size_t start = uri.rfind(png, 0) == 0 ? png.size() : uri.rfind(jpeg, 0) == 0 ? jpeg.size() : 0;
    if (!start)
        throw std::runtime_error("image_url must be a PNG/JPEG base64 data URI");
    size_t count = uri.size() - start;
    if (count == 0 || count % 4 || count > 12 * 1024 * 1024)
        throw std::runtime_error("invalid image base64 size");
    std::vector<uint8_t> bytes;
    bytes.reserve(count / 4 * 3);
    auto digit = [](char c) -> int {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        return -1;
    };
    for (size_t i = start; i < uri.size(); i += 4) {
        int a = digit(uri[i]), b = digit(uri[i + 1]), c = uri[i + 2] == '=' ? 0 : digit(uri[i + 2]),
            d = uri[i + 3] == '=' ? 0 : digit(uri[i + 3]);
        bool p2 = uri[i + 2] == '=', p3 = uri[i + 3] == '=';
        if (a < 0 || b < 0 || c < 0 || d < 0 || (p2 && !p3) ||
            ((p2 || p3) && i + 4 != uri.size()) || (p2 && (b & 15)) || (!p2 && p3 && (c & 3)))
            throw std::runtime_error("invalid image base64");
        unsigned value = (a << 18) | (b << 12) | (c << 6) | d;
        bytes.push_back(value >> 16);
        if (!p2)
            bytes.push_back((value >> 8) & 255);
        if (!p3)
            bytes.push_back(value & 255);
    }
    if (bytes.size() > 8 * 1024 * 1024)
        throw std::runtime_error("image exceeds 8MiB");
    bool is_png =
        bytes.size() > 8 && bytes[0] == 137 && bytes[1] == 80 && bytes[2] == 78 && bytes[3] == 71;
    bool is_jpeg = bytes.size() > 3 && bytes[0] == 255 && bytes[1] == 216 && bytes[2] == 255;
    if ((start == png.size() && !is_png) || (start == jpeg.size() && !is_jpeg))
        throw std::runtime_error("image MIME/signature mismatch");
    return bytes;
}
int main(int argc, char **argv) {
    try {
        std::string dir, host = "127.0.0.1", web = "web";
        int port = 8080;
        qwe::Qwen3Options options;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout
                    << "qwecpp_server --model DIR [--host 127.0.0.1] [--port 8080] [--web-root "
                       "web] [--devices 0] [--parallel pp|tp] [--kv-cache] [--max-context 2048]\n";
                return 0;
            }
            if (key == "--kv-cache") {
                options.kv_cache = true;
                continue;
            }
            if (i + 1 >= argc)
                throw std::runtime_error("missing argument");
            std::string value = argv[++i];
            if (key == "--model")
                dir = value;
            else if (key == "--host")
                host = value;
            else if (key == "--web-root")
                web = value;
            else if (key == "--devices")
                options.devices = qwe::parse_devices(value);
            else if (key == "--parallel") {
                if (value != "pp" && value != "tp")
                    throw std::runtime_error("server parallel must be pp or tp");
                options.tensor_parallel = value == "tp";
            } else if (key == "--port" || key == "--max-context") {
                size_t used;
                int n = std::stoi(value, &used);
                if (used != value.size())
                    throw std::runtime_error("invalid integer");
                if (key == "--port")
                    port = n;
                else
                    options.max_context = n;
            } else
                throw std::runtime_error("unknown option: " + key);
        }
        if (dir.empty() || port < 1 || port > 65535)
            throw std::runtime_error("model and valid port required");
        qwe::Tokenizer tokenizer(dir);
        qwe::TextModel model(dir, options);
        int device;
        qwe::check(cudaGetDevice(&device));
        if (!options.devices.empty())
            device = options.devices.front();
        auto ends = tokenizer.encode("<|im_end|>"), eos = tokenizer.encode("<|endoftext|>");
        if (ends.size() != 1 || eos.size() != 1)
            throw std::runtime_error("invalid special token IDs");
        std::mutex model_mutex, control_mutex;
        std::string active_id;
        std::shared_ptr<std::atomic<bool>> active_cancel;
        httplib::Server server;
        server.set_payload_max_length(16 * 1024 * 1024);
        server.set_read_timeout(15, 0);
        server.set_write_timeout(30, 0);
        server.set_default_headers(
            {{"X-Content-Type-Options", "nosniff"},
             {"Referrer-Policy", "same-origin"},
             {"Content-Security-Policy",
              "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src "
              "'self'; connect-src 'self'; frame-ancestors 'none'"}});
        if (!server.set_mount_point("/", web))
            throw std::runtime_error("web-root missing");
        const std::string model_name = std::filesystem::path(dir).filename().string();
        server.Get("/api/capabilities", [&](const httplib::Request &, httplib::Response &res) {
            Json j(cJSON_CreateObject(), cJSON_Delete);
            cJSON_AddStringToObject(j.get(), "model", model_name.c_str());
            cJSON_AddBoolToObject(j.get(), "vision", model.supports_vision());
            cJSON_AddNumberToObject(j.get(), "max_images", 4);
            cJSON_AddNumberToObject(j.get(), "max_image_bytes", 8 * 1024 * 1024);
            cJSON_AddNumberToObject(j.get(), "max_image_pixels", 262144);
            cJSON_AddBoolToObject(j.get(), "audio", false);
            cJSON_AddBoolToObject(j.get(), "stream", true);
            cJSON_AddBoolToObject(j.get(), "kv_cache", options.kv_cache);
            cJSON_AddNumberToObject(j.get(), "max_context", model.context_limit());
            cJSON_AddStringToObject(j.get(),
                                    "parallel",
                                    options.tensor_parallel       ? "tp"
                                    : options.devices.size() == 2 ? "pp"
                                                                  : "single");
            res.set_content(dump(j.get()), "application/json; charset=utf-8");
        });
        auto origin_ok = [](const httplib::Request &req) {
            auto origin = req.get_header_value("Origin");
            auto host = req.get_header_value("Host");
            return origin.empty() || origin == "http://" + host || origin == "https://" + host;
        };
        server.Post("/api/cancel", [&](const httplib::Request &req, httplib::Response &res) {
            if (!origin_ok(req)) {
                error(res, 403, "origin rejected");
                return;
            }
            auto j = parse(req.body);
            auto *id = cJSON_GetObjectItemCaseSensitive(j.get(), "id");
            if (!cJSON_IsString(id)) {
                error(res, 400, "id required");
                return;
            }
            std::lock_guard<std::mutex> guard(control_mutex);
            if (active_id == id->valuestring && active_cancel)
                active_cancel->store(true);
            res.set_content("{}", "application/json");
        });
        server.Post("/api/chat", [&](const httplib::Request &req, httplib::Response &res) {
            if (!origin_ok(req)) {
                error(res, 403, "origin rejected");
                return;
            }
            std::unique_lock<std::mutex> lock(model_mutex, std::try_to_lock);
            if (!lock) {
                error(res, 409, "模型正在生成，请稍后重试");
                return;
            }
            try {
                auto j = parse(req.body);
                auto *messages = cJSON_GetObjectItemCaseSensitive(j.get(), "messages"),
                     *id = cJSON_GetObjectItemCaseSensitive(j.get(), "id"),
                     *count = cJSON_GetObjectItemCaseSensitive(j.get(), "max_new_tokens");
                if (!cJSON_IsArray(messages) || !messages->child || !cJSON_IsString(id) ||
                    std::string(id->valuestring).size() > 64)
                    throw std::runtime_error("id and nonempty messages required");
                int maximum = 128;
                if (count) {
                    if (!cJSON_IsNumber(count) || count->valuedouble < 1 ||
                        count->valuedouble > 256 ||
                        std::floor(count->valuedouble) != count->valuedouble)
                        throw std::runtime_error("max_new_tokens must be 1..256");
                    maximum = (int)count->valuedouble;
                }
                auto begin = Clock::now();
                std::vector<qwe::ImageInput> image_inputs;
                double preprocess_ms = 0;
                size_t compressed_bytes = 0;
                std::string prompt;
                int index = 0;
                std::string last_role;
                for (auto *m = messages->child; m; m = m->next, ++index) {
                    auto *role = cJSON_GetObjectItemCaseSensitive(m, "role"),
                         *content = cJSON_GetObjectItemCaseSensitive(m, "content");
                    if (!cJSON_IsString(role))
                        throw std::runtime_error("message role required");
                    std::string r = role->valuestring;
                    if (r != "user" && r != "assistant" && !(r == "system" && index == 0))
                        throw std::runtime_error("invalid message role");
                    if (index && r == last_role)
                        throw std::runtime_error("messages must alternate");
                    last_role = r;
                    prompt += "<|im_start|>" + r + "\n";
                    if (cJSON_IsString(content))
                        prompt += content->valuestring;
                    else if (cJSON_IsArray(content) && content->child) {
                        for (auto *part = content->child; part; part = part->next) {
                            auto *type = cJSON_GetObjectItemCaseSensitive(part, "type");
                            if (!cJSON_IsString(type))
                                throw std::runtime_error("content part type required");
                            std::string kind = type->valuestring;
                            if (kind == "text") {
                                auto *text = cJSON_GetObjectItemCaseSensitive(part, "text");
                                if (!cJSON_IsString(text))
                                    throw std::runtime_error("text part must contain text");
                                prompt += text->valuestring;
                            } else if (kind == "image_url") {
                                if (r != "user" || !model.supports_vision())
                                    throw std::runtime_error(
                                        "this model/role does not support image input");
                                if (image_inputs.size() >= 4)
                                    throw std::runtime_error(
                                        "maximum four images across conversation");
                                auto *image = cJSON_GetObjectItemCaseSensitive(part, "image_url"),
                                     *url = cJSON_GetObjectItemCaseSensitive(image, "url");
                                if (!cJSON_IsString(url))
                                    throw std::runtime_error("image_url.url required");
                                auto start = Clock::now();
                                auto bytes = decode_image_uri(url->valuestring);
                                compressed_bytes += bytes.size();
                                if (compressed_bytes > 8 * 1024 * 1024)
                                    throw std::runtime_error("total compressed images exceed 8MiB");
                                image_inputs.push_back(qwe::preprocess_image(bytes));
                                preprocess_ms += ms(start, Clock::now());
                                const auto &img = image_inputs.back();
                                prompt += "<|vision_start|>";
                                for (int i = 0; i < img.grid_h * img.grid_w / 4; ++i)
                                    prompt += "<|image_pad|>";
                                prompt += "<|vision_end|>";
                            } else
                                throw std::runtime_error(
                                    "unsupported content part (text/image_url only)");
                        }
                    } else
                        throw std::runtime_error("content must be text or nonempty typed parts");
                    prompt += "<|im_end|>\n";
                }
                if (last_role != "user")
                    throw std::runtime_error("last message must be user");
                prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
                auto ids = tokenizer.encode(prompt);
                if (ids.size() >= (size_t)model.context_limit())
                    throw std::runtime_error("上下文超限，请新建会话或缩短输入");
                model.reset_cache();
                struct State {
                    std::unique_lock<std::mutex> lock;
                    std::vector<int> ids;
                    std::shared_ptr<std::atomic<bool>> cancel;
                    std::string id;
                    Clock::time_point begin;
                    size_t input;
                    std::vector<qwe::ImageInput> images;
                    double preprocess_ms;
                };
                auto state =
                    std::make_shared<State>(State{std::move(lock),
                                                  std::move(ids),
                                                  std::make_shared<std::atomic<bool>>(false),
                                                  id->valuestring,
                                                  begin,
                                                  0,
                                                  std::move(image_inputs),
                                                  preprocess_ms});
                state->input = state->ids.size();
                {
                    std::lock_guard<std::mutex> guard(control_mutex);
                    active_id = state->id;
                    active_cancel = state->cancel;
                }
                res.set_header("Cache-Control", "no-cache, no-transform");
                res.set_header("X-Accel-Buffering", "no");
                res.set_chunked_content_provider(
                    "text/event-stream; charset=utf-8",
                    [&, state, maximum](size_t, httplib::DataSink &sink) {
                        auto emit = [&](cJSON *j) {
                            auto payload = "data: " + dump(j) + "\n\n";
                            return sink.write(payload.data(), payload.size());
                        };
                        try {
                            qwe::DeviceGuard guard(device);
                            std::vector<qwe::ImageFeatures> image_features;
                            auto vision_start = Clock::now();
                            size_t image_tokens = 0;
                            for (const auto &image : state->images) {
                                if (state->cancel->load() || !sink.is_writable())
                                    break;
                                image_features.push_back(model.encode_image(image));
                                image_tokens += image.grid_h * image.grid_w / 4;
                            }
                            double vision_ms = ms(vision_start, Clock::now());
                            int generated = 0;
                            double ttft = -1, prefill = 0, decode = 0;
                            Clock::time_point first, last;
                            std::string pending, reason = "length";
                            for (; generated < maximum &&
                                   state->ids.size() < (size_t)model.context_limit();
                                 ++generated) {
                                if (state->cancel->load() || !sink.is_writable()) {
                                    reason = "cancelled";
                                    break;
                                }
                                auto start = Clock::now();
                                auto logits = image_features.empty()
                                                  ? model.forward(state->ids, true)
                                                  : model.forward_multimodal(
                                                        state->ids, image_features, true);
                                double elapsed = ms(start, Clock::now());
                                if (generated == 0)
                                    prefill += elapsed;
                                else
                                    decode += elapsed;
                                int token = (int)(std::max_element(logits.begin(), logits.end()) -
                                                  logits.begin());
                                if (token == ends[0] || token == eos[0]) {
                                    reason = "eos";
                                    break;
                                }
                                last = Clock::now();
                                if (!generated) {
                                    first = last;
                                    ttft = ms(state->begin, first);
                                }
                                state->ids.push_back(token);
                                pending += tokenizer.decode({token}, true);
                                auto text = drain(pending);
                                if (!text.empty()) {
                                    Json event(cJSON_CreateObject(), cJSON_Delete);
                                    cJSON_AddStringToObject(event.get(), "type", "delta");
                                    cJSON_AddStringToObject(event.get(), "text", text.c_str());
                                    if (!emit(event.get()))
                                        return false;
                                }
                            }
                            if (state->ids.size() >= (size_t)model.context_limit())
                                reason = "context";
                            Json done(cJSON_CreateObject(), cJSON_Delete);
                            cJSON_AddStringToObject(done.get(), "type", "done");
                            cJSON_AddStringToObject(done.get(), "finish_reason", reason.c_str());
                            auto *m = cJSON_AddObjectToObject(done.get(), "metrics");
                            double total = ms(state->begin, Clock::now());
                            cJSON_AddNumberToObject(m, "input_tokens", state->input);
                            cJSON_AddNumberToObject(m, "image_count", state->images.size());
                            cJSON_AddNumberToObject(m, "image_tokens", image_tokens);
                            cJSON_AddNumberToObject(m, "image_preprocess_ms", state->preprocess_ms);
                            cJSON_AddNumberToObject(m, "vision_ms", vision_ms);
                            cJSON_AddNumberToObject(m, "output_tokens", generated);
                            cJSON_AddNumberToObject(m, "ttft_ms", ttft);
                            cJSON_AddNumberToObject(
                                m,
                                "tpot_ms",
                                generated > 1 ? ms(first, last) / (generated - 1) : -1);
                            cJSON_AddNumberToObject(m, "total_ms", total);
                            cJSON_AddNumberToObject(
                                m, "output_tok_s", total > 0 ? generated * 1000 / total : 0);
                            cJSON_AddNumberToObject(m, "prefill_ms", prefill);
                            cJSON_AddNumberToObject(m, "decode_ms", decode);
                            cJSON_AddNumberToObject(m, "kv_cache_bytes", model.kv_cache_bytes());
                            emit(done.get());
                            sink.done();
                            return true;
                        } catch (const std::exception &e) {
                            Json event(cJSON_CreateObject(), cJSON_Delete);
                            cJSON_AddStringToObject(event.get(), "type", "error");
                            cJSON_AddStringToObject(event.get(), "error", e.what());
                            emit(event.get());
                            sink.done();
                            return true;
                        }
                    },
                    [&, state](bool) {
                        std::lock_guard<std::mutex> guard(control_mutex);
                        if (active_id == state->id) {
                            active_id.clear();
                            active_cancel.reset();
                        }
                    });
            } catch (const std::exception &e) {
                error(res, 400, e.what());
            }
        });
        std::cout << "QWECPP server: http://" << host << ":" << port << " model=" << model_name
                  << std::endl;
        if (!server.listen(host, port))
            throw std::runtime_error("listen failed");
    } catch (const std::exception &e) {
        std::cerr << "server: " << e.what() << "\n";
        return 1;
    }
}
