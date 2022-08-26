#pragma once
#define PCL_NO_PRECOMPILE
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct EIGEN_ALIGN16 PointA {
    PCL_ADD_POINT4D;
    double timestamp;
    std::uint16_t intensity;
    std::uint8_t flags;
    std::uint8_t elongation;
    std::uint16_t scan_id;
    std::uint16_t scan_idx;
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // pcl 1.9.1
    PCL_MAKE_ALIGNED_OPERATOR_NEW  // pcl 1.12
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointA,
    (float, x, x)(float, y, y)(float, z, z)(double, timestamp, timestamp)(
        std::uint16_t, intensity, intensity)(std::uint8_t, flags, flags)(
        std::uint8_t, elongation, elongation)(std::uint16_t, scan_id,
                                              scan_id)(std::uint16_t, scan_idx,
                                                       scan_idx));

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointCloud<PointType>::Ptr CloudPtr;

typedef std::pair<PointType, PointType> PointLine;
