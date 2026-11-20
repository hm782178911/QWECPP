#include "runtime/image_input.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_MAX_DIMENSIONS 4096
#include "stb_image.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace qwe {
static double cubic(double x) {
    x = std::abs(x);
    if (x < 1)
        return ((1.5 * x - 2.5) * x) * x + 1;
    if (x < 2)
        return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}
// separable bicubic抗锯齿；uint8中间结果，避免图像缩放引入无界数值。
static std::vector<uint8_t>
resize_axis(const std::vector<uint8_t> &src, int w, int h, int out, bool horizontal) {
    int in = horizontal ? w : h, ow = horizontal ? out : w, oh = horizontal ? h : out;
    std::vector<uint8_t> dst((size_t)ow * oh * 3);
    double scale = (double)in / out, filter = std::max(1.0, scale), support = 2 * filter;
    struct Filter {
        int start;
        std::vector<double> weights;
    };
    std::vector<Filter> filters;
    double maximum = 0;
    for (int x = 0; x < out; ++x) {
        double center = (x + .5) * scale;
        int start = std::max(0, (int)(center - support + .5)),
            end = std::min(in, (int)(center + support + .5));
        Filter f{start, std::vector<double>(end - start)};
        double total = 0;
        for (int i = start; i < end; ++i) {
            f.weights[i - start] = cubic((i + .5 - center) / filter);
            total += f.weights[i - start];
        }
        for (double &weight : f.weights) {
            weight /= total;
            maximum = std::max(maximum, weight);
        }
        filters.push_back(std::move(f));
    }
    // uint8 bicubic uses bounded signed fixed-point coefficients, matching the
    // torchvision CPU antialias algorithm (rather than a floating-point resize).
    unsigned precision = 0;
    for (; precision < 22; ++precision)
        if ((int)(.5 + maximum * (1 << (precision + 1))) >= (1 << 15))
            break;
    for (int x = 0; x < out; ++x) {
        auto &f = filters[x];
        std::vector<int> coefficients;
        for (double weight : f.weights) {
            double value = weight * (1 << precision);
            coefficients.push_back((int)(value < 0 ? value - .5 : value + .5));
        }
        for (int other = 0; other < (horizontal ? h : w); ++other)
            for (int c = 0; c < 3; ++c) {
                int sum = 1 << (precision - 1);
                for (size_t j = 0; j < coefficients.size(); ++j) {
                    int i = f.start + j;
                    size_t index = horizontal ? ((size_t)other * w + i) * 3 + c
                                              : ((size_t)i * w + other) * 3 + c;
                    sum += src[index] * coefficients[j];
                }
                size_t index = horizontal ? ((size_t)other * ow + x) * 3 + c
                                          : ((size_t)x * ow + other) * 3 + c;
                dst[index] = (uint8_t)std::clamp(sum >> precision, 0, 255);
            }
    }
    return dst;
}

ImageInput preprocess_image(const std::vector<uint8_t> &encoded) {
    if (encoded.empty() || encoded.size() > 8 * 1024 * 1024)
        throw std::runtime_error("image size must be 1..8MiB");
    bool png = encoded.size() >= 8 && encoded[0] == 137 && encoded[1] == 80 && encoded[2] == 78 &&
               encoded[3] == 71;
    bool jpg = encoded.size() >= 3 && encoded[0] == 255 && encoded[1] == 216 && encoded[2] == 255;
    if (!png && !jpg)
        throw std::runtime_error("only PNG/JPEG images supported");
    int w, h, channels;
    if (!stbi_info_from_memory(encoded.data(), (int)encoded.size(), &w, &h, &channels) || w < 1 ||
        h < 1 || w > 4096 || h > 4096 || (size_t)w * h > 16000000 ||
        std::max(w, h) > 200 * std::min(w, h))
        throw std::runtime_error("invalid image dimensions");
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
        stbi_load_from_memory(encoded.data(), (int)encoded.size(), &w, &h, &channels, 3),
        stbi_image_free);
    if (!decoded)
        throw std::runtime_error("image decode failed");
    int nh = std::max(32, (int)std::nearbyint(h / 32.) * 32),
        nw = std::max(32, (int)std::nearbyint(w / 32.) * 32);
    if (nh * nw > 262144) {
        double beta = std::sqrt((double)w * h / 262144);
        nh = std::max(32, (int)std::floor(h / beta / 32) * 32);
        nw = std::max(32, (int)std::floor(w / beta / 32) * 32);
    } else if (nh * nw < 65536) {
        double beta = std::sqrt(65536. / ((double)w * h));
        nh = (int)std::ceil(h * beta / 32) * 32;
        nw = (int)std::ceil(w * beta / 32) * 32;
    }
    if (nh * nw > 262144 || nh > 8192 || nw > 8192)
        throw std::runtime_error("unsupported image aspect/patch budget");
    std::vector<uint8_t> rgb(decoded.get(), decoded.get() + (size_t)w * h * 3);
    if (w != nw)
        rgb = resize_axis(rgb, w, h, nw, true);
    if (h != nh)
        rgb = resize_axis(rgb, nw, h, nh, false);
    int gh = nh / 16, gw = nw / 16;
    ImageInput result(gh, gw, w, h);
    for (int token = 0; token < gh * gw; ++token) {
        int block = token / 4, sub = token % 4, py = (block / (gw / 2)) * 2 + sub / 2,
            px = (block % (gw / 2)) * 2 + sub % 2;
        for (int c = 0; c < 3; ++c)
            for (int t = 0; t < 2; ++t)
                for (int y = 0; y < 16; ++y)
                    for (int x = 0; x < 16; ++x) {
                        int offset = ((c * 2 + t) * 16 + y) * 16 + x;
                        result.patches.data()[(size_t)token * 1536 + offset] =
                            (rgb[((size_t)(py * 16 + y) * nw + px * 16 + x) * 3 + c] / 255.f -
                             .5f) /
                            .5f;
                    }
    }
    return result;
}
} // namespace qwe
