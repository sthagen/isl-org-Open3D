// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tests/test_utility/Raw.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace open3d {
namespace tests {
// ----------------------------------------------------------------------------
// Raw data with SIZE = 1021 elements.
// ----------------------------------------------------------------------------
std::vector<uint8_t> Raw::data_ = {
        214, 100, 199, 203, 232, 50,  85,  195, 70,  141, 121, 160, 93,  130,
        242, 233, 162, 182, 36,  154, 4,   61,  34,  205, 39,  102, 33,  27,
        254, 55,  130, 213, 156, 75,  162, 133, 125, 248, 74,  196, 134, 196,
        102, 227, 72,  89,  205, 234, 17,  242, 134, 21,  49,  169, 227, 88,
        16,  5,   116, 16,  60,  247, 230, 216, 67,  137, 95,  193, 130, 170,
        135, 10,  111, 237, 237, 183, 72,  188, 163, 90,  175, 42,  112, 224,
        211, 84,  58,  227, 89,  175, 243, 150, 167, 218, 112, 235, 101, 207,
        174, 232, 123, 55,  242, 234, 37,  224, 163, 110, 157, 71,  200, 78,
        113, 57,  47,  70,  141, 106, 43,  231, 26,  32,  126, 193, 251, 238,
        174, 97,  191, 94,  75,  59,  149, 62,  38,  186, 31,  202, 41,  189,
        19,  242, 13,  132, 44,  61,  203, 186, 167, 246, 163, 193, 23,  34,
        132, 19,  17,  52,  117, 209, 146, 192, 13,  40,  254, 52,  226, 31,
        254, 13,  221, 18,  1,   235, 151, 45,  41,  99,  232, 209, 91,  140,
        147, 115, 175, 25,  135, 193, 77,  253, 147, 223, 190, 160, 9,   190,
        212, 235, 222, 211, 249, 189, 230, 250, 170, 126, 41,  211, 226, 19,
        165, 63,  160, 58,  178, 80,  83,  59,  18,  161, 57,  166, 130, 247,
        71,  139, 183, 28,  120, 151, 240, 114, 85,  216, 110, 0,   87,  152,
        212, 59,  172, 123, 122, 77,  181, 46,  158, 10,  105, 177, 171, 162,
        88,  47,  155, 159, 186, 83,  188, 51,  234, 174, 166, 65,  135, 22,
        66,  223, 174, 23,  28,  92,  147, 151, 169, 73,  197, 73,  84,  48,
        251, 0,   210, 84,  47,  111, 244, 234, 195, 178, 30,  174, 97,  197,
        240, 233, 219, 51,  202, 139, 75,  230, 231, 222, 127, 146, 41,  69,
        220, 125, 118, 216, 126, 74,  46,  174, 185, 35,  153, 125, 213, 184,
        45,  56,  127, 30,  35,  91,  82,  237, 231, 158, 213, 208, 126, 85,
        100, 168, 155, 66,  38,  18,  27,  165, 92,  73,  84,  23,  108, 238,
        148, 67,  167, 194, 124, 40,  225, 159, 132, 53,  142, 108, 211, 100,
        62,  83,  185, 162, 251, 86,  228, 34,  104, 1,   199, 197, 74,  29,
        220, 183, 12,  114, 251, 180, 53,  120, 220, 23,  25,  97,  76,  167,
        206, 33,  13,  13,  116, 199, 176, 112, 30,  150, 147, 135, 151, 92,
        77,  226, 121, 43,  155, 134, 157, 152, 59,  211, 17,  25,  235, 43,
        122, 57,  210, 74,  91,  223, 87,  207, 168, 9,   65,  198, 159, 213,
        78,  56,  50,  156, 27,  172, 199, 183, 51,  102, 80,  110, 58,  98,
        135, 39,  141, 3,   96,  97,  77,  188, 66,  165, 140, 234, 174, 206,
        177, 79,  164, 1,   135, 215, 157, 163, 132, 102, 92,  183, 204, 172,
        38,  8,   16,  174, 47,  157, 178, 144, 0,   1,   77,  66,  167, 218,
        46,  87,  170, 224, 166, 79,  225, 47,  40,  128, 211, 172, 230, 48,
        100, 180, 221, 139, 188, 237, 59,  236, 140, 238, 126, 140, 239, 203,
        207, 151, 167, 253, 238, 82,  222, 150, 162, 193, 197, 202, 67,  154,
        119, 42,  202, 220, 222, 169, 105, 156, 152, 164, 137, 37,  147, 8,
        178, 132, 212, 131, 28,  124, 130, 12,  207, 98,  162, 115, 36,  105,
        62,  103, 4,   182, 146, 207, 148, 113, 121, 253, 14,  18,  163, 152,
        56,  56,  160, 235, 188, 118, 111, 216, 242, 241, 229, 195, 85,  136,
        55,  121, 242, 118, 225, 246, 46,  116, 198, 195, 230, 65,  194, 245,
        84,  102, 143, 141, 158, 48,  121, 91,  166, 233, 53,  154, 220, 27,
        95,  50,  164, 151, 172, 152, 15,  143, 143, 61,  4,   87,  2,   235,
        153, 196, 226, 237, 44,  114, 124, 202, 162, 246, 39,  74,  224, 93,
        229, 190, 121, 69,  241, 31,  220, 158, 183, 235, 46,  71,  42,  51,
        159, 44,  32,  58,  241, 3,   41,  30,  117, 165, 233, 25,  156, 17,
        100, 126, 111, 74,  62,  232, 144, 48,  8,   110, 207, 192, 90,  254,
        9,   133, 51,  168, 178, 83,  226, 164, 87,  12,  195, 204, 178, 173,
        230, 79,  191, 75,  206, 48,  150, 13,  25,  40,  62,  34,  150, 14,
        226, 241, 14,  235, 119, 65,  149, 43,  149, 121, 207, 236, 134, 148,
        185, 57,  67,  161, 137, 4,   237, 88,  52,  133, 102, 78,  173, 164,
        113, 68,  179, 84,  54,  193, 65,  174, 4,   215, 217, 153, 81,  170,
        134, 216, 63,  65,  18,  131, 226, 155, 135, 209, 244, 187, 87,  91,
        11,  6,   1,   124, 74,  180, 209, 129, 119, 19,  48,  123, 235, 11,
        21,  62,  181, 155, 23,  245, 221, 42,  121, 193, 198, 1,   147, 187,
        189, 235, 24,  200, 241, 25,  70,  61,  206, 24,  190, 70,  44,  239,
        194, 24,  250, 215, 87,  176, 116, 110, 166, 82,  153, 33,  20,  96,
        34,  168, 29,  224, 148, 53,  170, 134, 79,  240, 195, 31,  9,   131,
        101, 53,  115, 40,  78,  110, 1,   165, 32,  117, 21,  198, 200, 174,
        232, 221, 15,  12,  134, 45,  236, 27,  98,  152, 162, 178, 137, 103,
        209, 147, 234, 56,  201, 95,  97,  24,  206, 98,  190, 238, 216, 212,
        182, 161, 131, 159, 128, 147, 171, 7,   192, 152, 35,  36,  49,  198,
        215, 187, 46,  170, 79,  26,  226, 26,  122, 68,  50,  73,  167, 241,
        56,  129, 198, 238, 36,  75,  143, 164, 222, 59,  171, 160, 212, 207,
        197, 7,   150, 157, 194, 197, 72,  19,  224, 44,  45,  91,  112, 96,
        165, 25,  83,  221, 154, 26,  205, 191, 101, 93,  100, 69,  152, 17,
        229, 110, 224, 172, 117, 120, 74,  57,  62,  147, 76,  32,  191, 122,
        123, 49,  219, 34,  74,  47,  0,   229, 73,  206, 165, 175, 44,  11,
        244, 197, 28,  219, 52,  253, 136, 170, 118, 211, 227, 181, 103, 49,
        213, 39,  171, 82,  88,  135, 116, 163, 182, 117, 137, 1,   68,  48,
        176, 113, 60,  166, 55,  89,  131, 108, 87,  12,  24,  206, 224};

// ----------------------------------------------------------------------------
// Get the next uint8_t value.
// Output range: [0, 255].
// ----------------------------------------------------------------------------
template <>
uint8_t Raw::Next() {
    uint8_t output = data_[index];
    index = (index + step) % SIZE;

    return output;
}

// ----------------------------------------------------------------------------
// Get the next int value.
// Output range: [0, 255].
// ----------------------------------------------------------------------------
template <>
int Raw::Next() {
    int output = (int)data_[index];
    index = (index + step) % SIZE;

    return output;
}

// ----------------------------------------------------------------------------
// Get the next int value.
// Output range: [0, 255].
// ----------------------------------------------------------------------------
template <>
size_t Raw::Next() {
    size_t output = (size_t)data_[index];
    index = (index + step) % SIZE;

    return output;
}

// ----------------------------------------------------------------------------
// Get the next float value.
// Output range: [0, 1].
// ----------------------------------------------------------------------------
template <>
float Raw::Next() {
    float output = (float)data_[index] / VMAX;
    index = (index + step) % SIZE;

    return output;
}

// ----------------------------------------------------------------------------
// Get the next double value.
// Output range: [0, 1].
// ----------------------------------------------------------------------------
template <>
double Raw::Next() {
    double output = (double)data_[index] / VMAX;
    index = (index + step) % SIZE;

    return output;
}

}  // namespace tests
}  // namespace open3d
