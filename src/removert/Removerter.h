#pragma once
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "../innolog/logger.h"
#include "../common/type.h"

using std::cout;
using std::endl;

struct SphericalPoint {
    float az;  // azimuth
    float el;  // elevation
    float r;   // radius
};

class Removerter {
   private:
    int kNumOmpCores = 16;
    const float kFlagNoPOINT = 0.f;
    const float kValidDiffUpperBound = 200.0;

    // Static sensitivity
    int kNumKnnPointsToCompare;  // static sensitivity (increase this value,
                                 // less static structure will be removed at the
                                 // scan-side removal stage)
    float kScanKnnAndMapKnnAvgDiffThreshold;  // static sensitivity (decrease
                                              // this value, less static
                                              // structure will be removed at
                                              // the scan-side removal stage)
    double kSubmapResolution = 20;
    float kDownsampleVoxelSize = 0.02;
    bool kFlagSaveMapPointcloud = true;
    bool kFlagSaveCleanScans = true;
    size_t kMinVoxelPoints = 1;

    // Eigen::Matrix4d kSE3MatExtrinsicLiDARtoPoseBase =
    //     Eigen::Matrix4d::Identity();
    // Eigen::Matrix4d kSE3MatExtrinsicPoseBasetoLiDAR =
    //     Eigen::Matrix4d::Identity();

    // Point Cloud Input directories
    std::string sequence_scan_dir_;
    std::vector<std::string> sequence_scan_names_;
    std::vector<std::string> sequence_scan_paths_;

    // Point Cloud Output directories
    std::string save_pcd_directory_;
    std::string scan_static_save_dir_;
    std::string scan_dynamic_save_dir_;
    std::string map_static_save_dir_;
    std::string map_dynamic_save_dir_;
    std::string depth_scan_save_dir_;
    std::string diff_map_save_dir_;
    std::string depth_map_save_dir_;
    std::string scan_csv_save_dir_;
    std::string diff_csv_save_dir_;
    std::string map_csv_save_dir_;
    std::string ground_save_dir_;
    std::string noground_save_dir_;
    // Sequence pose file
    std::string sequence_pose_path_;
    std::vector<Eigen::Matrix4d> sequence_scan_poses_;
    std::vector<Eigen::Matrix4d> sequence_scan_inverse_poses_;

    // Memory allocated
    pcl::PointCloud<PointType>::Ptr map_global_orig_;
    // the M_i. i.e., removert is: M1 -> S1 + D1, D1 -> M2 , M2 -> S2 + D2 ...
    // repeat ...
    pcl::PointCloud<PointType>::Ptr map_global_curr_;
    pcl::PointCloud<PointType>::Ptr map_local_curr_;
    pcl::PointCloud<PointType>::Ptr map_global_curr_static_;   // the S_i
    pcl::PointCloud<PointType>::Ptr map_global_curr_dynamic_;  // the D_i
    pcl::PointCloud<PointType>::Ptr map_subset_global_curr_;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_map_global_curr_;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_scan_global_curr_;
    // Downsampled pointcloud
    size_t start_id_ = 0;
    size_t end_id_ = 3000;
    std::vector<pcl::PointCloud<PointType>::Ptr> scans_;
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<Eigen::Matrix4d> inverse_poses_;
    std::vector<std::string> sequence_valid_scan_names_;
    std::vector<std::string> sequence_valid_scan_paths_;
    std::vector<pcl::PointCloud<PointType>::Ptr> scans_static_;
    std::vector<pcl::PointCloud<PointType>::Ptr> scans_dynamic_;

    // removert params
    float kVFOV;
    float kHFOV;
    std::pair<float, float> kFOV;
    std::vector<float> remove_resolution_list_;
    std::vector<float> revert_resolution_list_;

    float curr_res_alpha_;  // just for tracking current status

    const int base_node_idx_ = 0;

    // NOT recommend to use for under 5 million points map input (becausing
    // not-using is just faster)
    const bool kUseSubsetMapCloud = true;
    const float kBallSize = 200.0;  // meter
    const float kRangeLimit = 255.f;

   public:
    Removerter();
    ~Removerter();
    void run(void);
    void allocateMemory();

    void parseValidScanInfo();
    void readValidScans();

    void makeGlobalMap();
    void mergeScansWithinGlobalCoord(
        const std::vector<pcl::PointCloud<PointType>::Ptr>& _scans,
        const std::vector<Eigen::Matrix4d>& _scans_poses,
        pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save);
    void octreeDownsampling(const pcl::PointCloud<PointType>::Ptr& _src,
                            pcl::PointCloud<PointType>::Ptr& _to_save);
    void voxelDownsampling(const pcl::PointCloud<PointType>::Ptr& _src,
                            pcl::PointCloud<PointType>::Ptr& _to_save);
    void submap_voxelDownsampling(const pcl::PointCloud<PointType>::Ptr& _src,
                            pcl::PointCloud<PointType>::Ptr& _to_save);

    void transformGlobalMapToLocal(int _base_scan_idx);
    void transformGlobalMapToLocal(int _base_scan_idx,
                                   pcl::PointCloud<PointType>::Ptr& _map_local);
    void transformGlobalMapToLocal(
        const pcl::PointCloud<PointType>::Ptr& _map_global, int _base_scan_idx,
        pcl::PointCloud<PointType>::Ptr& _map_local);

    void removeOnce(float _res);
    void revertOnce(float _res);

    std::vector<int> calcDescrepancyAndParseDynamicPointIdxForEachScan(
        std::pair<int, int> _rimg_shape);
    void parseDynamicMapPointcloudUsingPtIdx(std::vector<int>& _point_indexes);
    void parseStaticMapPointcloudUsingPtIdx(std::vector<int>& _point_indexes);
    std::vector<int> getGlobalMapStaticIdxFromDynamicIdx(
        const std::vector<int>& _dynamic_point_indexes);
    std::vector<int> getStaticIdxFromDynamicIdx(
        const std::vector<int>& _dynamic_point_indexes, int _num_all_points);

    // void saveCurrentStaticMapHistory(
    //     void);  // the 0th element is a noisy (original input) (actually not
    //             // static) map.
    // void saveCurrentDynamicMapHistory(void);

    void takeGlobalMapSubsetWithinBall(int _center_scan_idx);
    void transformGlobalMapSubsetToLocal(int _base_scan_idx);

    cv::Mat scan2RangeImg(
        const pcl::PointCloud<PointType>::Ptr& _scan,
        const std::pair<float, float>
            _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
        const std::pair<int, int> _rimg_size);
    std::pair<cv::Mat, cv::Mat> map2RangeImg(
        const pcl::PointCloud<PointType>::Ptr& _scan,
        const std::pair<float, float>
            _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
        const std::pair<int, int> _rimg_size);

    std::vector<int> calcDescrepancyAndParseDynamicPointIdx(
        const cv::Mat& _scan_rimg, const cv::Mat& _diff_rimg,
        const cv::Mat& _map_rimg_ptidx);
    std::vector<int> selectDescrepancyAndParseDynamicPointIdx(
        const cv::Mat& _scan_rimg, const cv::Mat& _map_rimg,
        const cv::Mat& _map_rimg_ptidx);

    void parsePointcloudSubsetUsingPtIdx(
        const pcl::PointCloud<PointType>::Ptr& _ptcloud_orig,
        std::vector<int>& _point_indexes,
        pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save);
    void parseMapPointcloudSubsetUsingPtIdx(
        std::vector<int>& _point_indexes,
        pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save);

    void saveCurrentStaticAndDynamicPointCloudGlobal(void);
    void saveCurrentStaticAndDynamicPointCloudLocal(int _base_pose_idx = 0);

    pcl::PointCloud<PointType>::Ptr local2global(
        const pcl::PointCloud<PointType>::Ptr& _scan_local, int _scan_idx);
    pcl::PointCloud<PointType>::Ptr global2local(
        const pcl::PointCloud<PointType>::Ptr& _scan_global, int _scan_idx);

    // scan-side removal
    std::pair<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr>
    removeDynamicPointsOfScanByKnn(int _scan_idx);
    void removeDynamicPointsAndSaveStaticScanForEachScan(void);

    void scansideRemovalForEachScan(void);
    void saveCleanedScans(void);
    void saveMapPointcloudByMergingCleanedScans(void);
    void scansideRemovalForEachScanAndSaveThem(void);

    void saveStaticScan(int _scan_idx,
                        const pcl::PointCloud<PointType>::Ptr& _ptcloud);
    void saveDynamicScan(int _scan_idx,
                         const pcl::PointCloud<PointType>::Ptr& _ptcloud);

};  // Removerter

std::pair<int, int> resetRimgSize(const std::pair<float, float> _fov,
                                  const float _resize_ratio);
SphericalPoint cart2sph(const PointType& _cp);

template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N - 1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) *x = val;
    return xs;
}