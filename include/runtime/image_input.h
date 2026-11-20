#pragma once
#include "core/tensor.hpp"
#include <vector>
#include <cstdint>
namespace qwe {
/** 原生RGB图片预处理结果：patches为[grid_h*grid_w,1536]，排列按2x2 merge块优先。
 * 静态帧复制为2帧，RGB归一化到[-1,1]；HostTensor持有内存。 */
struct ImageInput {
    int grid_h, grid_w, source_width, source_height;
    HostTensor patches;
    ImageInput(int gh, int gw, int sw, int sh)
        : grid_h(gh), grid_w(gw), source_width(sw), source_height(sh), patches({gh * gw, 1536}) {}
};
// 只接受PNG/JPEG，压缩数据最多8MiB，解码最多4096x4096及1600万像素。
// bicubic抗锯齿resize，32倍数网格，65536..262144像素；失败抛异常。
ImageInput preprocess_image(const std::vector<uint8_t> &encoded);
} // namespace qwe
