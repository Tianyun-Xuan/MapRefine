#include "Removerter.h"

#include "../ground/ground_segmentation.h"
inline float rad2deg(float radians) { return radians * 180.0 / M_PI; }

inline float deg2rad(float degrees) { return degrees * M_PI / 180.0; }

void fsmkdir(std::string _path) {
    if (!std::filesystem::is_directory(_path) ||
        !std::filesystem::exists(_path))
        std::filesystem::create_directories(_path);  // create src folder
}  // fsmkdir

Removerter::Removerter() {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    // using higher, more strict static
    kNumKnnPointsToCompare = 5;
    // using smaller, more strict static
    kScanKnnAndMapKnnAvgDiffThreshold = 0.05;
    kVFOV = 25;
    kHFOV = 120;
    kFOV = std::pair<float, float>(kVFOV, kHFOV);
    sequence_scan_dir_ = "/home/xavier/repos/floam/data/lidar/";
    sequence_pose_path_ = "/home/xavier/repos/MapRefine/pose.txt";
    save_pcd_directory_ = "/home/xavier/repos/MapRefine/data/";
    /*
    Range image resolution
    the below is actually magnifier ratio (the x1 means 1 deg x 1 deg per pixel)
    the first removing resolution's magnifier ratio
    should meet the seonsor vertical fov / number of rays
    HDL 64E of KITTI dataset
    -> appx 25 deg / 64 ray ~ 0.4 deg per pixel -> the magnifier ratio = 1/0.4
    = 2.5
    Ouster OS1-64 of MulRan dataset
    -> appx 45 deg / 64 ray ~ 0.7 deg per pixel -> the magnifier ratio = 1/0.7
    = 1.4
    recommend to use the first reverting resolution's magnifier ratio
    should lied in 1.0 to 1.5
    remove_resolution_list: [2.5, 2.0, 1.5] # for HDL 64E of KITTI dataset
    */
    remove_resolution_list_ = {10};
    if (save_pcd_directory_.substr(save_pcd_directory_.size() - 1, 1) !=
        std::string("/"))
        save_pcd_directory_ = save_pcd_directory_ + "/";
    fsmkdir(save_pcd_directory_);
    scan_static_save_dir_ = save_pcd_directory_ + "scan_static";
    fsmkdir(scan_static_save_dir_);
    scan_dynamic_save_dir_ = save_pcd_directory_ + "scan_dynamic";
    fsmkdir(scan_dynamic_save_dir_);
    map_static_save_dir_ = save_pcd_directory_ + "map_static";
    fsmkdir(map_static_save_dir_);
    map_dynamic_save_dir_ = save_pcd_directory_ + "map_dynamic";
    fsmkdir(map_dynamic_save_dir_);
    depth_map_save_dir_ = save_pcd_directory_ + "depth_map";
    fsmkdir(depth_map_save_dir_);
    diff_map_save_dir_ = save_pcd_directory_ + "diff_map";
    fsmkdir(diff_map_save_dir_);
    depth_scan_save_dir_ = save_pcd_directory_ + "depth_scan";
    fsmkdir(depth_scan_save_dir_);
    ground_save_dir_ = save_pcd_directory_ + "ground";
    fsmkdir(ground_save_dir_);
    noground_save_dir_ = save_pcd_directory_ + "no_ground";
    fsmkdir(noground_save_dir_);

    // scan_csv_save_dir_ = save_pcd_directory_ + "scan_csv";
    // fsmkdir(scan_csv_save_dir_);
    // diff_csv_save_dir_ = save_pcd_directory_ + "diff_csv";
    // fsmkdir(diff_csv_save_dir_);
    // map_csv_save_dir_ = save_pcd_directory_ + "map_csv";
    // fsmkdir(map_csv_save_dir_);
    allocateMemory();
}  // ctor

void Removerter::allocateMemory() {
    map_global_orig_.reset(new pcl::PointCloud<PointType>());
    map_global_curr_.reset(new pcl::PointCloud<PointType>());
    map_local_curr_.reset(new pcl::PointCloud<PointType>());
    map_global_curr_static_.reset(new pcl::PointCloud<PointType>());
    map_global_curr_dynamic_.reset(new pcl::PointCloud<PointType>());
    map_subset_global_curr_.reset(new pcl::PointCloud<PointType>());
    kdtree_map_global_curr_.reset(new pcl::KdTreeFLANN<PointType>());
    kdtree_scan_global_curr_.reset(new pcl::KdTreeFLANN<PointType>());
    // what about using swap here ?
    map_global_orig_->clear();
    map_global_curr_->clear();
    map_local_curr_->clear();
    map_global_curr_static_->clear();
    map_global_curr_dynamic_->clear();
    map_subset_global_curr_->clear();
}  // allocateMemory

Removerter::~Removerter() {}

void Removerter::parseValidScanInfo(void) {
    for (auto& item : std::filesystem::directory_iterator(sequence_scan_dir_)) {
        std::filesystem::path temp = item;
        std::filesystem::path filename = temp.stem();
        if (temp.extension() == ".pcd") {
            sequence_scan_names_.emplace_back(filename);
            sequence_scan_paths_.emplace_back(temp);
        }
    }
    // Should keep the same size and order
    std::sort(sequence_scan_names_.begin(), sequence_scan_names_.end());
    std::sort(sequence_scan_paths_.begin(), sequence_scan_paths_.end());

    std::ifstream tum_pose(sequence_pose_path_);
    long long ts;
    double tx, ty, tz, qx, qy, qz, qw;
    while (tum_pose >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
        Eigen::Quaterniond tq(qw, qx, qy, qz);
        Eigen::Vector3d tt(tx, ty, tz);
        Eigen::Matrix4d trt = Eigen::Matrix4d::Identity();
        trt.block(0, 0, 3, 3) = tq.toRotationMatrix();
        trt.block(0, 3, 3, 1) = tt;
        sequence_scan_poses_.emplace_back(trt);
        sequence_scan_inverse_poses_.emplace_back(trt.inverse());
    }
    inno_log_info(
        "Scan name size: %zu, Scan path size: %zu, Scan pose size: %zu, "
        "Scan inverse pose size: %zu",
        sequence_scan_names_.size(), sequence_scan_poses_.size(),
        sequence_scan_paths_.size(), sequence_scan_inverse_poses_.size());

}  // parseValidScanInfo

void Removerter::readValidScans(void) {
    std::shared_ptr<GroundSegmentation> ground_segmentation =
        std::make_shared<GroundSegmentation>();
    // No-flyback model
    for (size_t i = start_id_; i < end_id_; i += 2) {
        pcl::PointCloud<PointA>::Ptr raw_points(new pcl::PointCloud<PointA>);
        pcl::io::loadPCDFile<PointA>(sequence_scan_paths_[i], *raw_points);
        pcl::PointCloud<PointType>::Ptr points(new pcl::PointCloud<PointType>);
        for (auto& point : raw_points->points) {
            float curr_range =
                sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (curr_range < kRangeLimit) {
                PointType temp{point.z, -point.y, point.x};
                temp.intensity = point.intensity;
                points->emplace_back(temp);
            }
        }
        // remove ground
        pcl::PointCloud<PointType>::Ptr ground(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr no_ground(
            new pcl::PointCloud<PointType>);
        ground_segmentation->segment(points, ground, no_ground);
        std::string static_ground_file_name =
            ground_save_dir_ + "/" + std::to_string(i) + ".pcd";
        std::string static_noground_file_name =
            noground_save_dir_ + "/" + std::to_string(i) + ".pcd";
        if (ground->size())
            pcl::io::savePCDFileBinary(static_ground_file_name, *ground);
        if (no_ground->size())
            pcl::io::savePCDFileBinary(static_noground_file_name, *no_ground);
        // pcdown
        pcl::VoxelGrid<PointType> downsize_filter;
        downsize_filter.setLeafSize(kDownsampleVoxelSize, kDownsampleVoxelSize,
                                    kDownsampleVoxelSize);
        downsize_filter.setInputCloud(no_ground);

        pcl::PointCloud<PointType>::Ptr downsampled_points(
            new pcl::PointCloud<PointType>);
        downsize_filter.filter(*downsampled_points);

        // save downsampled pointcloud
        scans_.emplace_back(downsampled_points);
        poses_.emplace_back(sequence_scan_poses_[i]);
        inverse_poses_.emplace_back(sequence_scan_inverse_poses_[i]);
        sequence_valid_scan_names_.emplace_back(sequence_scan_names_[i]);
        sequence_valid_scan_paths_.emplace_back(sequence_scan_paths_[i]);

        // // Debug
        // inno_log_info(
        //     "Pointcloud size: %zu, downsample size: %zu, current scans size:
        //     "
        //     "%zu",
        //     points->points.size(), downsampled_points->points.size(),
        //     scans_.size());
    }
}  // readValidScans

void Removerter::makeGlobalMap(void) {
    // transform local to global and merging the scans
    map_global_orig_->clear();
    map_global_curr_->clear();
    mergeScansWithinGlobalCoord(scans_, poses_, map_global_orig_);

    inno_log_info("Map pointcloud size: %zu", map_global_orig_->points.size());
    inno_log_info("Downsampling leaf size: %f", kDownsampleVoxelSize);
    // remove repeated (redundant) points
    // - using OctreePointCloudVoxelCentroid for large-size downsampling
    pcl::PointCloud<PointType>::Ptr map_voxel_temp(
        new pcl::PointCloud<PointType>());
    voxelDownsampling(map_global_orig_, map_voxel_temp);
    octreeDownsampling(map_voxel_temp, map_global_curr_);

    // save the original cloud
    if (kFlagSaveMapPointcloud) {
        // in global coord
        std::string static_global_file_name =
            save_pcd_directory_ + "OriginalNoisyMapGlobal.pcd";
        std::string static_voxel_file_name =
            save_pcd_directory_ + "VoxelMapGlobal.pcd";
        pcl::io::savePCDFileBinary(static_global_file_name, *map_global_curr_);
        pcl::io::savePCDFileBinary(static_voxel_file_name, *map_voxel_temp);
        inno_log_info("The original pointcloud is saved (global coord): %s",
                      static_global_file_name.c_str());
    }
    // make tree (for fast ball search for the projection to make a map range
    // image later) NOT recommend to use for under 5 million points map input
    if (kUseSubsetMapCloud) {
        kdtree_map_global_curr_->setInputCloud(map_global_curr_);
    }
}  // makeGlobalMap

void Removerter::voxelDownsampling(const pcl::PointCloud<PointType>::Ptr& _src,
                                   pcl::PointCloud<PointType>::Ptr& _to_save) {
    double minX, minY, minZ, maxX, maxY, maxZ;
    PointType min_pt;
    PointType max_pt;
    pcl::getMinMax3D(*_src, min_pt, max_pt);
    float minValue = std::numeric_limits<float>::epsilon() * 512.0f;
    minX = min_pt.x;
    minY = min_pt.y;
    minZ = min_pt.z;
    maxX = max_pt.x + minValue;
    maxY = max_pt.y + minValue;
    maxZ = max_pt.z + minValue;
    inno_log_info("X Y Z [%f, %f] [%f, %f] [%f, %f]", minX, maxX, minY, maxY,
                  minZ, maxZ);

    int sx, sy, sz;
    sx = std::ceil((maxX - minX) / kSubmapResolution) + 1;
    sy = std::ceil((maxY - minY) / kSubmapResolution) + 1;
    sz = std::ceil((maxZ - minZ) / kSubmapResolution) + 1;
    int total_submap = sx * sy * sz;
    inno_log_info("%u %u %u", sx, sy, sz);
    inno_log_info("%u", total_submap);

    std::vector<pcl::PointCloud<PointType>::Ptr> submaps;
    for (int i = 0; i < total_submap; ++i) {
        pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>);
        submaps.emplace_back(temp);
    }
    inno_log_info("submaps size %zu", submaps.size());
    inno_log_info("submap init point size %zu", submaps[0]->size());

    for (auto& point : _src->points) {
        int px, py, pz;
        px = std::round((point.x - minX) / kSubmapResolution);
        py = std::round((point.y - minY) / kSubmapResolution);
        pz = std::round((point.z - minZ) / kSubmapResolution);
        int pid = pz * sy * sx + py * sx + px;
        submaps[pid]->push_back(point);
    }
    inno_log_info("submap done");

    // Killed lol
    // std::unordered_map<size_t, std::vector<size_t>> voxelID_pointID;
    // std::vector<Eigen::MatrixXi> voxel_flag(rx, Eigen::MatrixXi(ry,rz));
    for (auto& pc : submaps) {
        if (pc->size()) {
            submap_voxelDownsampling(pc, _to_save);
        }
    }
}
void Removerter::submap_voxelDownsampling(
    const pcl::PointCloud<PointType>::Ptr& _src,
    pcl::PointCloud<PointType>::Ptr& _to_save) {
    double minX, minY, minZ, maxX, maxY, maxZ;
    PointType min_pt;
    PointType max_pt;
    pcl::getMinMax3D(*_src, min_pt, max_pt);
    float minValue = std::numeric_limits<float>::epsilon() * 512.0f;
    minX = min_pt.x;
    minY = min_pt.y;
    minZ = min_pt.z;
    maxX = max_pt.x + minValue;
    maxY = max_pt.y + minValue;
    maxZ = max_pt.z + minValue;
    inno_log_info("sub map X Y Z [%f, %f] [%f, %f] [%f, %f]", minX, maxX, minY,
                  maxY, minZ, maxZ);

    int rx, ry, rz;
    rx = std::ceil((maxX - minX) / kDownsampleVoxelSize) + 1;
    ry = std::ceil((maxY - minY) / kDownsampleVoxelSize) + 1;
    rz = std::ceil((maxZ - minZ) / kDownsampleVoxelSize) + 1;
    int total_cell = rx * ry * rz;

    std::unordered_map<int, std::vector<size_t>> voxelID_pointID;
    for (size_t i = 0; i < _src->size(); ++i) {
        auto point = _src->points[i];
        int tx = std::round((point.x - minX) / kDownsampleVoxelSize);
        int ty = std::round((point.y - minY) / kDownsampleVoxelSize);
        int tz = std::round((point.z - minZ) / kDownsampleVoxelSize);
        int id = tz * ry * rx + ty * rx + tx;
        if (auto iter = voxelID_pointID.find(id);
            iter != voxelID_pointID.end()) {
            iter->second.emplace_back(i);
        } else {
            voxelID_pointID.insert({id, {i}});
        }
    }
    inno_log_info("%u %u %u", rx, ry, rz);
    inno_log_info("%u", total_cell);
    inno_log_info("Static Done %zu", voxelID_pointID.size());

    for (auto& pair : voxelID_pointID) {
        if (pair.second.size() >= kMinVoxelPoints) {
            for (auto& index : pair.second) {
                _to_save->push_back(_src->points[index]);
            }
        }
    }
}

void Removerter::mergeScansWithinGlobalCoord(
    const std::vector<pcl::PointCloud<PointType>::Ptr>& _scans,
    const std::vector<Eigen::Matrix4d>& _scans_poses,
    pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save) {
    inno_log_info("Scan size: %zu, Pose size: %zu", _scans.size(),
                  _scans_poses.size());
    // NOTE: _scans must be in local coord
    for (std::size_t scan_idx = 0; scan_idx < _scans.size(); ++scan_idx) {
        pcl::PointCloud<PointType>::Ptr ii_scan = _scans.at(scan_idx);
        Eigen::Matrix4d ii_pose = _scans_poses.at(scan_idx);
        // local to global (local2global)
        pcl::PointCloud<PointType>::Ptr scan_global_coord(
            new pcl::PointCloud<PointType>());
        // pcl::transformPointCloud(*ii_scan, *scan_global_coord,
        //                          kSE3MatExtrinsicLiDARtoPoseBase);
        pcl::transformPointCloud(*ii_scan, *scan_global_coord, ii_pose);
        // merge the scan into the global map
        *_ptcloud_to_save += *scan_global_coord;
    }
}  // mergeScansWithinGlobalCoord

void Removerter::octreeDownsampling(const pcl::PointCloud<PointType>::Ptr& _src,
                                    pcl::PointCloud<PointType>::Ptr& _to_save) {
    pcl::octree::OctreePointCloudVoxelCentroid<PointType> octree(
        kDownsampleVoxelSize);
    octree.setInputCloud(_src);
    octree.defineBoundingBox();
    octree.addPointsFromInputCloud();
    pcl::octree::OctreePointCloudVoxelCentroid<PointType>::AlignedPointTVector
        centroids;
    octree.getVoxelCentroids(centroids);

    // init current map with the downsampled full cloud
    _to_save->points.assign(centroids.begin(), centroids.end());
    _to_save->width = 1;
    _to_save->height =
        _to_save->points.size();  // make sure again the format of the
                                  // downsampled point cloud
    inno_log_info("Downsampled pointcloud size: %zu", _to_save->points.size());
}  // octreeDownsampling

void Removerter::removeOnce(float _res_alpha) {
    // filter spec (i.e., a shape of the range image)
    curr_res_alpha_ = _res_alpha;

    std::pair<int, int> rimg_shape = resetRimgSize(kFOV, _res_alpha);
    float deg_per_pixel = 1.0 / _res_alpha;
    inno_log_info("Removing starts with resolution: x %f(%fdeg/pixel",
                  _res_alpha, deg_per_pixel);
    inno_log_info("The range image size is: [%u, %u]", rimg_shape.first,
                  rimg_shape.second);
    inno_log_info("The number of map points: %zu",
                  map_global_curr_->points.size());
    inno_log_info("-- ... starts cleaning ... ");

    // map-side removal: remove and get dynamic (will be removed) points'index
    // set
    std::vector<int> dynamic_point_indexes =
        calcDescrepancyAndParseDynamicPointIdxForEachScan(rimg_shape);
    inno_log_info("-- The number of dynamic points: %zu",
                  dynamic_point_indexes.size());
    parseDynamicMapPointcloudUsingPtIdx(dynamic_point_indexes);

    // static_point_indexes == complemently indexing dynamic_point_indexes
    std::vector<int> static_point_indexes =
        getGlobalMapStaticIdxFromDynamicIdx(dynamic_point_indexes);
    inno_log_info("-- The number of static points: %zu",
                  static_point_indexes.size());
    parseStaticMapPointcloudUsingPtIdx(static_point_indexes);

    // Update the current map and reset the tree
    map_global_curr_->clear();
    *map_global_curr_ = *map_global_curr_static_;

    // // NOT recommend to use for under 5 million points
    if (kUseSubsetMapCloud) {
        kdtree_map_global_curr_->setInputCloud(map_global_curr_);
    }

}  // removeOnce

std::vector<int> Removerter::calcDescrepancyAndParseDynamicPointIdxForEachScan(
    std::pair<int, int> _rimg_shape) {
    std::vector<int> dynamic_point_indexes;
    // dynamic_point_indexes.reserve(100000);
    for (std::size_t idx_scan = 0; idx_scan < scans_.size(); ++idx_scan) {
        // curr scan
        pcl::PointCloud<PointType>::Ptr _scan = scans_.at(idx_scan);
        // scan's pointcloud to range img
        cv::Mat scan_rimg =
            scan2RangeImg(_scan, kFOV, _rimg_shape);  // openMP inside
        cv::Mat bit8;
        scan_rimg.convertTo(bit8, CV_8UC1);
        cv::imwrite(depth_scan_save_dir_ + "/" + std::to_string(idx_scan) +
                        "_" + std::to_string(_rimg_shape.first) + "x" +
                        std::to_string(_rimg_shape.second) + ".tiff",
                    bit8);
        // std::ofstream fs;
        // fs.open(scan_csv_save_dir_ + "/" + std::to_string(idx_scan) +
        // ".csv"); if (fs.is_open()) {
        //     fs << cv::format(scan_rimg, cv::Formatter::FormatType::FMT_CSV);
        //     fs.close();
        // } else {
        //     inno_log_error("Failed to open csv output");
        // }

        // map's pointcloud to range img
        if (kUseSubsetMapCloud) {
            takeGlobalMapSubsetWithinBall(idx_scan);
            // the most time comsuming part 1
            transformGlobalMapSubsetToLocal(idx_scan);
        } else {
            // if the input map size (of a batch) is short, just using this line
            // is more fast.
            // - e.g., 100-1000m or ~5 million points are ok, empirically more
            // than 10Hz
            transformGlobalMapToLocal(idx_scan);
        }
        auto [map_rimg, map_rimg_ptidx] =
            map2RangeImg(map_local_curr_, kFOV,
                         _rimg_shape);  // the most time comsuming part 2 ->so
                                        // openMP applied inside

        // diff range img
        const int kNumRimgRow = _rimg_shape.first;
        const int kNumRimgCol = _rimg_shape.second;
        cv::Mat diff_rimg =
            cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1,
                    cv::Scalar::all(0.0));  // float matrix, save range value
        cv::absdiff(scan_rimg, map_rimg, diff_rimg);
        map_rimg.convertTo(bit8, CV_8UC1);
        cv::imwrite(depth_map_save_dir_ + "/" + std::to_string(idx_scan) + "_" +
                        std::to_string(_rimg_shape.first) + "x" +
                        std::to_string(_rimg_shape.second) + ".tiff",
                    bit8);
        // fs.open(map_csv_save_dir_ + "/" + std::to_string(idx_scan) + ".csv");
        // if (fs.is_open()) {
        //     fs << cv::format(map_rimg, cv::Formatter::FormatType::FMT_CSV);
        //     fs.close();
        // } else {
        //     inno_log_error("Failed to open csv output");
        // }
        diff_rimg.convertTo(bit8, CV_8UC1);
        cv::imwrite(diff_map_save_dir_ + "/" + std::to_string(idx_scan) + "_" +
                        std::to_string(_rimg_shape.first) + "x" +
                        std::to_string(_rimg_shape.second) + ".tiff",
                    bit8);
        // fs.open(diff_csv_save_dir_ + "/" + std::to_string(idx_scan) +
        // ".csv"); if (fs.is_open()) {
        //     fs << cv::format(diff_rimg, cv::Formatter::FormatType::FMT_CSV);
        //     fs.close();
        // } else {
        //     inno_log_error("Failed to open csv output");
        // }

        // parse dynamic points' indexes: rule: If a pixel value of diff_rimg is
        // larger, scan is the further - means that pixel of submap is likely
        // dynamic.
        std::vector<int> this_scan_dynamic_point_indexes =
            selectDescrepancyAndParseDynamicPointIdx(scan_rimg, map_rimg,
                                                     map_rimg_ptidx);
        dynamic_point_indexes.insert(dynamic_point_indexes.end(),
                                     this_scan_dynamic_point_indexes.begin(),
                                     this_scan_dynamic_point_indexes.end());
    }  // for_each scan Done

    // remove repeated indexes
    std::set<int> dynamic_point_indexes_set(dynamic_point_indexes.begin(),
                                            dynamic_point_indexes.end());
    std::vector<int> dynamic_point_indexes_unique(
        dynamic_point_indexes_set.begin(), dynamic_point_indexes_set.end());

    return dynamic_point_indexes_unique;
}  // calcDescrepancyForEachScan

cv::Mat Removerter::scan2RangeImg(
    const pcl::PointCloud<PointType>::Ptr& _scan,
    const std::pair<float, float>
        _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
    const std::pair<int, int> _rimg_size) {
    const float kVFOV = _fov.first;
    const float kHFOV = _fov.second;

    const int kNumRimgRow = _rimg_size.first;
    const int kNumRimgCol = _rimg_size.second;
    // cout << "rimg size is: [" << _rimg_size.first << ", " <<
    // _rimg_size.second << "]." << endl;

    // @ range image initizliation
    cv::Mat rimg = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1,
                           cv::Scalar::all(kFlagNoPOINT));  // float matrix

    // @ points to range img
    int num_points = _scan->points.size();
#pragma omp parallel for num_threads(kNumOmpCores)
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx) {
        PointType this_point = _scan->points[pt_idx];
        SphericalPoint sph_point = cart2sph(this_point);

        // @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0,30]
        // @ min and max is just for the easier (naive) boundary checks.
        int lower_bound_row_idx{0};
        int lower_bound_col_idx{0};
        int upper_bound_row_idx{kNumRimgRow - 1};
        int upper_bound_col_idx{kNumRimgCol - 1};
        int pixel_idx_row = int(std::min(
            std::max(std::round(kNumRimgRow * (1 - (rad2deg(sph_point.el) +
                                                    (kVFOV / float(2.0))) /
                                                       (kVFOV - float(0.0)))),
                     float(lower_bound_row_idx)),
            float(upper_bound_row_idx)));
        int pixel_idx_col = int(
            std::min(std::max(std::round(kNumRimgCol * ((rad2deg(sph_point.az) +
                                                         (kHFOV / float(2.0))) /
                                                        (kHFOV - float(0.0)))),
                              float(lower_bound_col_idx)),
                     float(upper_bound_col_idx)));

        float curr_range = sph_point.r;

        // @ Theoretically, this if-block would have race condition (i.e.,this
        // is a critical section),
        // @ But, the resulting range image is acceptable (watching via Rviz),
        // @      so I just naively applied omp pragma for this whole for-block
        // (2020.10.28)
        // @ Reason: because this for loop is splited by the omp, points in a
        // single splited for range do not race among them,
        // @         also, a point A and B lied in different for-segments do not
        // tend to correspond to the same pixel, #so we can
        // assume practically there are few race conditions.
        // @ P.S. some explicit mutexing directive makes the code even slower
        // ref:https://stackoverflow.com/questions/2396430/how-to-use-lock-in-openmp
        if (rimg.at<float>(pixel_idx_row, pixel_idx_col) <= 0.f ||
            curr_range < rimg.at<float>(pixel_idx_row, pixel_idx_col)) {
            rimg.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
        }
    }

    return rimg;
}  // scan2RangeImg

std::pair<cv::Mat, cv::Mat> Removerter::map2RangeImg(
    const pcl::PointCloud<PointType>::Ptr& _scan,
    const std::pair<float, float>
        _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
    const std::pair<int, int> _rimg_size) {
    const float kVFOV = _fov.first;
    const float kHFOV = _fov.second;

    const int kNumRimgRow = _rimg_size.first;
    const int kNumRimgCol = _rimg_size.second;

    // @ range image initizliation
    cv::Mat rimg = cv::Mat(
        kNumRimgRow, kNumRimgCol, CV_32FC1,
        cv::Scalar::all(kFlagNoPOINT));  // float matrix, save range value
    cv::Mat rimg_ptidx = cv::Mat(
        kNumRimgRow, kNumRimgCol, CV_32SC1,
        cv::Scalar::all(0));  // int matrix, save point (of global map) index

    // @ points to range img
    int num_points = _scan->points.size();
#pragma omp parallel for num_threads(kNumOmpCores)
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx) {
        PointType this_point = _scan->points[pt_idx];
        SphericalPoint sph_point = cart2sph(this_point);

        // @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0,30]
        // @ min and max is just for the easier (naive) boundary checks.
        int lower_bound_row_idx{0};
        int lower_bound_col_idx{0};
        int upper_bound_row_idx{kNumRimgRow - 1};
        int upper_bound_col_idx{kNumRimgCol - 1};
        int pixel_idx_row = int(std::min(
            std::max(std::round(kNumRimgRow * (1 - (rad2deg(sph_point.el) +
                                                    (kVFOV / float(2.0))) /
                                                       (kVFOV - float(0.0)))),
                     float(lower_bound_row_idx)),
            float(upper_bound_row_idx)));
        int pixel_idx_col = int(
            std::min(std::max(std::round(kNumRimgCol * ((rad2deg(sph_point.az) +
                                                         (kHFOV / float(2.0))) /
                                                        (kHFOV - float(0.0)))),
                              float(lower_bound_col_idx)),
                     float(upper_bound_col_idx)));

        float curr_range = sph_point.r;

        // @ Theoretically, this if-block would have race condition (i.e.,this
        // is a critical section),
        // @ But, the resulting range image is acceptable (watching via Rviz),
        // @      so I just naively applied omp pragma for this whole for-block
        // (2020.10.28)
        // @ Reason: because this for loop is splited by the omp, points in a
        // single splited for range do not race among them,
        // @         also, a point A and B lied in different for-segments do not
        // tend to correspond to the same pixel, #               so we can
        // assume practically there are few race conditions.
        // @ P.S. some explicit mutexing directive makes the code even slower
        // ref:https://stackoverflow.com/questions/2396430/how-to-use-lock-in-openmp
        if (curr_range > rimg.at<float>(pixel_idx_row, pixel_idx_col)) {
            rimg.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
            rimg_ptidx.at<int>(pixel_idx_row, pixel_idx_col) = pt_idx;
        }
    }

    return std::pair<cv::Mat, cv::Mat>(rimg, rimg_ptidx);
}  // map2RangeImg

void Removerter::transformGlobalMapToLocal(int _base_scan_idx) {
    Eigen::Matrix4d base_pose_inverse = inverse_poses_.at(_base_scan_idx);
    // global to local (global2local)
    map_local_curr_->clear();
    pcl::transformPointCloud(*map_global_curr_, *map_local_curr_,
                             base_pose_inverse);
    // pcl::transformPointCloud(*map_local_curr_, *map_local_curr_,
    //                          kSE3MatExtrinsicPoseBasetoLiDAR);
}  // transformGlobalMapToLocal
void Removerter::transformGlobalMapToLocal(
    int _base_scan_idx, pcl::PointCloud<PointType>::Ptr& _map_local) {
    Eigen::Matrix4d base_pose_inverse = inverse_poses_.at(_base_scan_idx);
    // global to local (global2local)
    _map_local->clear();
    pcl::transformPointCloud(*map_global_curr_, *_map_local, base_pose_inverse);
    // pcl::transformPointCloud(*_map_local, *_map_local,
    //                          kSE3MatExtrinsicPoseBasetoLiDAR);
}  // transformGlobalMapToLocal
void Removerter::transformGlobalMapToLocal(
    const pcl::PointCloud<PointType>::Ptr& _map_global, int _base_scan_idx,
    pcl::PointCloud<PointType>::Ptr& _map_local) {
    Eigen::Matrix4d base_pose_inverse = inverse_poses_.at(_base_scan_idx);
    // global to local (global2local)
    _map_local->clear();
    pcl::transformPointCloud(*_map_global, *_map_local, base_pose_inverse);
    // pcl::transformPointCloud(*_map_local, *_map_local,
    //                          kSE3MatExtrinsicPoseBasetoLiDAR);
}  // transformGlobalMapToLocal

void Removerter::transformGlobalMapSubsetToLocal(int _base_scan_idx) {
    Eigen::Matrix4d base_pose_inverse = inverse_poses_.at(_base_scan_idx);
    // global to local (global2local)
    map_local_curr_->clear();
    pcl::transformPointCloud(*map_subset_global_curr_, *map_local_curr_,
                             base_pose_inverse);
    // pcl::transformPointCloud(*map_local_curr_, *map_local_curr_,
    //                          kSE3MatExtrinsicPoseBasetoLiDAR);

}  // transformGlobalMapSubsetToLocal

void Removerter::parseMapPointcloudSubsetUsingPtIdx(
    std::vector<int>& _point_indexes,
    pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save) {
    // extractor
    pcl::ExtractIndices<PointType> extractor;
    std::shared_ptr<std::vector<int>> index_ptr =
        std::make_shared<std::vector<int>>(_point_indexes);
    extractor.setInputCloud(map_global_curr_);
    extractor.setIndices(index_ptr);
    extractor.setNegative(false);  // If set to true, you can extract point
                                   // clouds outside the specified index

    // parse
    _ptcloud_to_save->clear();
    extractor.filter(*_ptcloud_to_save);
}  // parseMapPointcloudSubsetUsingPtIdx

void Removerter::parseStaticMapPointcloudUsingPtIdx(
    std::vector<int>& _point_indexes) {
    // extractor
    pcl::ExtractIndices<PointType> extractor;
    std::shared_ptr<std::vector<int>> index_ptr =
        std::make_shared<std::vector<int>>(_point_indexes);
    extractor.setInputCloud(map_global_curr_);
    extractor.setIndices(index_ptr);
    extractor.setNegative(false);  // If set to true, you can extract point
                                   // clouds outside the specified index

    // parse
    map_global_curr_static_->clear();
    extractor.filter(*map_global_curr_static_);
}  // parseStaticMapPointcloudUsingPtIdx

void Removerter::parseDynamicMapPointcloudUsingPtIdx(
    std::vector<int>& _point_indexes) {
    // extractor
    pcl::ExtractIndices<PointType> extractor;
    std::shared_ptr<std::vector<int>> index_ptr =
        std::make_shared<std::vector<int>>(_point_indexes);
    extractor.setInputCloud(map_global_curr_);
    extractor.setIndices(index_ptr);
    extractor.setNegative(false);  // If set to true, you can extract point
                                   // clouds outside the specified index

    // parse
    map_global_curr_dynamic_->clear();
    extractor.filter(*map_global_curr_dynamic_);
}  // parseDynamicMapPointcloudUsingPtIdx

void Removerter::saveCurrentStaticAndDynamicPointCloudGlobal(void) {
    if (!kFlagSaveMapPointcloud) return;
    std::string curr_res_alpha_str = std::to_string(curr_res_alpha_);
    // dynamic
    std::string dyna_file_name = map_dynamic_save_dir_ +
                                 "/DynamicMapMapsideGlobalResX" +
                                 curr_res_alpha_str + ".pcd";
    pcl::io::savePCDFileBinary(dyna_file_name, *map_global_curr_dynamic_);
    // static
    std::string static_file_name = map_static_save_dir_ +
                                   "/StaticMapMapsideGlobalResX" +
                                   curr_res_alpha_str + ".pcd";
    pcl::io::savePCDFileBinary(static_file_name, *map_global_curr_static_);
}  // saveCurrentStaticAndDynamicPointCloudGlobal

void Removerter::saveCurrentStaticAndDynamicPointCloudLocal(
    int _base_node_idx) {
    if (!kFlagSaveMapPointcloud) return;

    std::string curr_res_alpha_str = std::to_string(curr_res_alpha_);

    // dynamic
    pcl::PointCloud<PointType>::Ptr map_local_curr_dynamic(
        new pcl::PointCloud<PointType>);
    transformGlobalMapToLocal(map_global_curr_dynamic_, _base_node_idx,
                              map_local_curr_dynamic);
    std::string dyna_file_name = map_dynamic_save_dir_ +
                                 "/DynamicMapMapsideLocalResX" +
                                 curr_res_alpha_str + ".pcd";
    pcl::io::savePCDFileBinary(dyna_file_name, *map_local_curr_dynamic);

    // static
    pcl::PointCloud<PointType>::Ptr map_local_curr_static(
        new pcl::PointCloud<PointType>);
    transformGlobalMapToLocal(map_global_curr_static_, _base_node_idx,
                              map_local_curr_static);
    std::string static_file_name = map_static_save_dir_ +
                                   "/StaticMapMapsideLocalResX" +
                                   curr_res_alpha_str + ".pcd";
    pcl::io::savePCDFileBinary(static_file_name, *map_local_curr_static);

}  // saveCurrentStaticAndDynamicPointCloudLocal

std::vector<int> Removerter::calcDescrepancyAndParseDynamicPointIdx(
    const cv::Mat& _scan_rimg, const cv::Mat& _diff_rimg,
    const cv::Mat& _map_rimg_ptidx) {
    // int num_dyna_points{0};
    // TODO: tracking the number of dynamic-assigned points and decide
    // when to stop removing (currently just fixed iteration e.g.,
    // [2.5, 2.0, 1.5])

    std::vector<int> dynamic_point_indexes;
    for (int row_idx = 0; row_idx < _diff_rimg.rows; row_idx++) {
        for (int col_idx = 0; col_idx < _diff_rimg.cols; col_idx++) {
            float this_diff = _diff_rimg.at<float>(row_idx, col_idx);
            float this_range = _scan_rimg.at<float>(row_idx, col_idx);
            // meter, // i.e., if 4m apart point, it should be 0.4m

            // float adaptive_coeff = 0.02;
            // // be diff (nearer) wrt the query
            // float adaptive_dynamic_descrepancy_threshold =
            //     std::fmin(adaptive_coeff * this_range,
            //               0.1f);  // adaptive descrepancy threshold
            float adaptive_dynamic_descrepancy_threshold = 0.1;

            // exclude no-point pixels either on scan img or map img (100 is
            // roughly 100 meter)
            if (this_diff < kValidDiffUpperBound &&
                this_diff >
                    adaptive_dynamic_descrepancy_threshold /* dynamic */) {
                // dynamic
                int this_point_idx_in_global_map =
                    _map_rimg_ptidx.at<int>(row_idx, col_idx);
                dynamic_point_indexes.emplace_back(
                    this_point_idx_in_global_map);

                // num_dyna_points++; // TODO
            }
        }
    }

    return dynamic_point_indexes;
}  // calcDescrepancyAndParseDynamicPointIdx

std::vector<int> Removerter::selectDescrepancyAndParseDynamicPointIdx(
    const cv::Mat& _scan_rimg, const cv::Mat& _map_rimg,
    const cv::Mat& _map_rimg_ptidx) {
    std::vector<int> dynamic_point_indexes;
    for (int row_idx = 0; row_idx < _scan_rimg.rows; row_idx++) {
        for (int col_idx = 0; col_idx < _scan_rimg.cols; col_idx++) {
            float scan_range = _scan_rimg.at<float>(row_idx, col_idx);
            float map_range = _map_rimg.at<float>(row_idx, col_idx);

            // float adaptive_coeff = 0.02;
            // // be diff (nearer) wrt the query
            // float adaptive_dynamic_descrepancy_threshold =
            //     std::fmin(adaptive_coeff * this_range,
            //               0.1f);  // adaptive descrepancy threshold
            float adaptive_dynamic_descrepancy_threshold = 0.01;

            // exclude no-point pixels either on scan img or map img (100 is
            // roughly 100 meter)
            if (scan_range != map_range) {
                // dynamic
                int this_point_idx_in_global_map =
                    _map_rimg_ptidx.at<int>(row_idx, col_idx);
                dynamic_point_indexes.emplace_back(
                    this_point_idx_in_global_map);
            }
        }
    }

    return dynamic_point_indexes;
}

void Removerter::takeGlobalMapSubsetWithinBall(int _center_scan_idx) {
    Eigen::Matrix4d center_pose_se3 = poses_.at(_center_scan_idx);
    PointType center_pose;
    center_pose.x = float(center_pose_se3(0, 3));
    center_pose.y = float(center_pose_se3(1, 3));
    center_pose.z = float(center_pose_se3(2, 3));

    std::vector<int> subset_indexes;
    std::vector<float> pointSearchSqDisGlobalMap;
    kdtree_map_global_curr_->radiusSearch(
        center_pose, kRangeLimit, subset_indexes, pointSearchSqDisGlobalMap, 0);
    parseMapPointcloudSubsetUsingPtIdx(subset_indexes, map_subset_global_curr_);
}  // takeMapSubsetWithinBall

std::vector<int> Removerter::getStaticIdxFromDynamicIdx(
    const std::vector<int>& _dynamic_point_indexes, int _num_all_points) {
    std::vector<int> pt_idx_all =
        linspace<int>(0, _num_all_points, _num_all_points);

    std::set<int> pt_idx_all_set(pt_idx_all.begin(), pt_idx_all.end());
    for (auto& _dyna_pt_idx : _dynamic_point_indexes) {
        pt_idx_all_set.erase(_dyna_pt_idx);
    }

    std::vector<int> static_point_indexes(pt_idx_all_set.begin(),
                                          pt_idx_all_set.end());
    return static_point_indexes;
}  // getStaticIdxFromDynamicIdx

std::vector<int> Removerter::getGlobalMapStaticIdxFromDynamicIdx(
    const std::vector<int>& _dynamic_point_indexes) {
    int num_all_points = map_global_curr_->points.size();
    return getStaticIdxFromDynamicIdx(_dynamic_point_indexes, num_all_points);
}  // getGlobalMapStaticIdxFromDynamicIdx

// void Removerter::saveCurrentStaticMapHistory(void) {
//     // deep copy
//     pcl::PointCloud<PointType>::Ptr map_global_curr_static(
//         new pcl::PointCloud<PointType>);
//     *map_global_curr_static = *map_global_curr_;

//     // save
//     static_map_global_history_.emplace_back(map_global_curr_static);
// }  // saveCurrentStaticMapHistory

// void Removerter::revertOnce(float _res_alpha) {
//     std::pair<int, int> rimg_shape = resetRimgSize(kFOV, _res_alpha);
//     float deg_per_pixel = 1.0 / _res_alpha;
//     inno_log_info("Reverting starts with resolution: x %f(%fdeg/pixel",
//                   _res_alpha, deg_per_pixel);
//     inno_log_info("The range image size is: [%u, %u]", rimg_shape.first,
//                   rimg_shape.second);
//     inno_log_info("The number of map points: %zu",
//                   map_global_curr_->points.size());
//     inno_log_info("-- ... starts cleaning ... ");
//     // TODO

// }  // revertOnce

void Removerter::parsePointcloudSubsetUsingPtIdx(
    const pcl::PointCloud<PointType>::Ptr& _ptcloud_orig,
    std::vector<int>& _point_indexes,
    pcl::PointCloud<PointType>::Ptr& _ptcloud_to_save) {
    // extractor
    pcl::ExtractIndices<PointType> extractor;
    std::shared_ptr<std::vector<int>> index_ptr =
        std::make_shared<std::vector<int>>(_point_indexes);
    extractor.setInputCloud(_ptcloud_orig);
    extractor.setIndices(index_ptr);
    extractor.setNegative(false);  // If set to true, you can extract point
                                   // clouds outside the specified index

    // parse
    _ptcloud_to_save->clear();
    extractor.filter(*_ptcloud_to_save);
}  // parsePointcloudSubsetUsingPtIdx

pcl::PointCloud<PointType>::Ptr Removerter::local2global(
    const pcl::PointCloud<PointType>::Ptr& _scan_local, int _scan_idx) {
    Eigen::Matrix4d scan_pose = poses_.at(_scan_idx);

    pcl::PointCloud<PointType>::Ptr scan_global(
        new pcl::PointCloud<PointType>());
    // pcl::transformPointCloud(*_scan_local, *scan_global,
    //                          kSE3MatExtrinsicLiDARtoPoseBase);
    pcl::transformPointCloud(*_scan_local, *scan_global, scan_pose);
    return scan_global;
}

pcl::PointCloud<PointType>::Ptr Removerter::global2local(
    const pcl::PointCloud<PointType>::Ptr& _scan_global, int _scan_idx) {
    Eigen::Matrix4d base_pose_inverse = inverse_poses_.at(_scan_idx);

    pcl::PointCloud<PointType>::Ptr scan_local(
        new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*_scan_global, *scan_local, base_pose_inverse);
    // pcl::transformPointCloud(*scan_local, *scan_local,
    //                          kSE3MatExtrinsicPoseBasetoLiDAR);

    return scan_local;
}

std::pair<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr>
Removerter::removeDynamicPointsOfScanByKnn(int _scan_idx) {
    // curr scan (in local coord)
    pcl::PointCloud<PointType>::Ptr scan_orig = scans_.at(_scan_idx);
    // auto scan_pose = poses_.at(_scan_idx);

    // curr scan (in global coord)
    pcl::PointCloud<PointType>::Ptr scan_orig_global =
        local2global(scan_orig, _scan_idx);
    int num_points_of_a_scan = scan_orig_global->points.size();
    if (!num_points_of_a_scan)
        inno_log_error(
            "[removeDynamicPointsOfScanByKnn] scan_orig_global size null");
    kdtree_scan_global_curr_->setInputCloud(scan_orig_global);

    //
    pcl::PointCloud<PointType>::Ptr scan_static_global(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr scan_dynamic_global(
        new pcl::PointCloud<PointType>);
    for (auto pt_idx = 0; pt_idx < num_points_of_a_scan; pt_idx++) {
        std::vector<int> topk_indexes_scan;
        std::vector<float> topk_L2dists_scan;
        kdtree_scan_global_curr_->nearestKSearch(
            scan_orig_global->points[pt_idx], kNumKnnPointsToCompare,
            topk_indexes_scan, topk_L2dists_scan);
        float sum_topknn_dists_in_scan =
            accumulate(topk_L2dists_scan.begin(), topk_L2dists_scan.end(), 0.0);
        float avg_topknn_dists_in_scan =
            sum_topknn_dists_in_scan / float(kNumKnnPointsToCompare);

        std::vector<int> topk_indexes_map;
        std::vector<float> topk_L2dists_map;
        kdtree_map_global_curr_->nearestKSearch(
            scan_orig_global->points[pt_idx], kNumKnnPointsToCompare,
            topk_indexes_map, topk_L2dists_map);
        float sum_topknn_dists_in_map =
            accumulate(topk_L2dists_map.begin(), topk_L2dists_map.end(), 0.0);
        float avg_topknn_dists_in_map =
            sum_topknn_dists_in_map / float(kNumKnnPointsToCompare);

        //
        if (std::abs(avg_topknn_dists_in_scan - avg_topknn_dists_in_map) <
            kScanKnnAndMapKnnAvgDiffThreshold) {
            scan_static_global->push_back(scan_orig_global->points[pt_idx]);
        } else {
            scan_dynamic_global->push_back(scan_orig_global->points[pt_idx]);
        }
    }

    // again global2local because later in the merging global map function,
    // which requires scans within each local coord.
    pcl::PointCloud<PointType>::Ptr scan_static_local =
        global2local(scan_static_global, _scan_idx);
    pcl::PointCloud<PointType>::Ptr scan_dynamic_local =
        global2local(scan_dynamic_global, _scan_idx);

    inno_log_info("The scan %s",
                  sequence_valid_scan_paths_.at(_scan_idx).c_str());
    inno_log_info("-- The number of static points in a scan: %zu",
                  scan_static_local->points.size());
    inno_log_info(" -- The number of dynamic points in a scan: %zu",
                  num_points_of_a_scan - scan_static_local->points.size());

    return std::pair<pcl::PointCloud<PointType>::Ptr,
                     pcl::PointCloud<PointType>::Ptr>(scan_static_local,
                                                      scan_dynamic_local);

}  // removeDynamicPointsOfScanByKnn

void Removerter::saveStaticScan(
    int _scan_idx, const pcl::PointCloud<PointType>::Ptr& _ptcloud) {
    std::string file_name_orig = sequence_valid_scan_names_.at(_scan_idx);
    std::string file_name =
        scan_static_save_dir_ + "/" + file_name_orig + ".pcd";
    inno_log_info("Scan %u 's static points is saved %s", _scan_idx,
                  file_name.c_str());
    pcl::io::savePCDFileBinary(file_name, *_ptcloud);
}  // saveStaticScan

void Removerter::saveDynamicScan(
    int _scan_idx, const pcl::PointCloud<PointType>::Ptr& _ptcloud) {
    std::string file_name_orig = sequence_valid_scan_names_.at(_scan_idx);
    std::string file_name =
        scan_dynamic_save_dir_ + "/" + file_name_orig + ".pcd";
    inno_log_info("Scan %u 's dynamic points is saved %s", _scan_idx,
                  file_name.c_str());
    pcl::io::savePCDFileBinary(file_name, *_ptcloud);
}  // saveDynamicScan

void Removerter::saveCleanedScans(void) {
    if (!kFlagSaveCleanScans) return;

    for (std::size_t idx_scan = 0; idx_scan < scans_static_.size();
         idx_scan++) {
        saveStaticScan(idx_scan, scans_static_.at(idx_scan));
        saveDynamicScan(idx_scan, scans_dynamic_.at(idx_scan));
    }
}  // saveCleanedScans

void Removerter::saveMapPointcloudByMergingCleanedScans(void) {
    // merge for verification
    if (!kFlagSaveMapPointcloud) return;

    // static map
    {
        pcl::PointCloud<PointType>::Ptr
            map_global_static_scans_merged_to_verify_full(
                new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr
            map_global_static_scans_merged_to_verify(
                new pcl::PointCloud<PointType>);
        mergeScansWithinGlobalCoord(
            scans_static_, poses_,
            map_global_static_scans_merged_to_verify_full);
        octreeDownsampling(map_global_static_scans_merged_to_verify_full,
                           map_global_static_scans_merged_to_verify);

        // global
        std::string global_file_name =
            map_static_save_dir_ + "/StaticMapScansideMapGlobal.pcd";
        pcl::io::savePCDFileBinary(global_file_name,
                                   *map_global_static_scans_merged_to_verify);
        inno_log_info(
            "[For verification] A static pointcloud (cleaned scans merged) is "
            "saved (global coord): %s",
            global_file_name.c_str());

        // local
        pcl::PointCloud<PointType>::Ptr map_local_static_scans_merged_to_verify(
            new pcl::PointCloud<PointType>);
        int base_node_idx = base_node_idx_;
        transformGlobalMapToLocal(map_global_static_scans_merged_to_verify,
                                  base_node_idx,
                                  map_local_static_scans_merged_to_verify);
        std::string local_file_name =
            map_static_save_dir_ + "/StaticMapScansideMapLocal.pcd";
        pcl::io::savePCDFileBinary(local_file_name,
                                   *map_local_static_scans_merged_to_verify);
        inno_log_info(
            "[For verification] A static pointcloud (cleaned scans merged) is "
            "saved (local coord)): %s",
            local_file_name.c_str());
    }

    // dynamic map
    {
        pcl::PointCloud<PointType>::Ptr
            map_global_dynamic_scans_merged_to_verify_full(
                new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr
            map_global_dynamic_scans_merged_to_verify(
                new pcl::PointCloud<PointType>);
        mergeScansWithinGlobalCoord(
            scans_dynamic_, poses_,
            map_global_dynamic_scans_merged_to_verify_full);
        octreeDownsampling(map_global_dynamic_scans_merged_to_verify_full,
                           map_global_dynamic_scans_merged_to_verify);

        // global
        std::string global_file_name =
            map_dynamic_save_dir_ + "/DynamicMapScansideMapGlobal.pcd";
        pcl::io::savePCDFileBinary(global_file_name,
                                   *map_global_dynamic_scans_merged_to_verify);
        inno_log_info(
            "[For verification] A dynamic pointcloud (cleaned scans merged) is "
            "saved (global coord): %s",
            global_file_name.c_str());

        // local
        pcl::PointCloud<PointType>::Ptr
            map_local_dynamic_scans_merged_to_verify(
                new pcl::PointCloud<PointType>);
        int base_node_idx = base_node_idx_;
        transformGlobalMapToLocal(map_global_dynamic_scans_merged_to_verify,
                                  base_node_idx,
                                  map_local_dynamic_scans_merged_to_verify);
        std::string local_file_name =
            map_dynamic_save_dir_ + "/DynamicMapScansideMapLocal.pcd";
        pcl::io::savePCDFileBinary(local_file_name,
                                   *map_local_dynamic_scans_merged_to_verify);
        inno_log_info(
            "[For verification] A dynamic pointcloud (cleaned scans merged) is "
            "saved (local coord)): %s",
            local_file_name.c_str());
    }
}  // saveMapPointcloudByMergingCleanedScans

void Removerter::scansideRemovalForEachScan(void) {
    // for fast scan-side neighbor search
    kdtree_map_global_curr_->setInputCloud(map_global_curr_);

    // for each scan
    for (std::size_t idx_scan = 0; idx_scan < scans_.size(); idx_scan++) {
        auto [this_scan_static, this_scan_dynamic] =
            removeDynamicPointsOfScanByKnn(idx_scan);
        scans_static_.emplace_back(this_scan_static);
        scans_dynamic_.emplace_back(this_scan_dynamic);
    }
}  // scansideRemovalForEachScan

void Removerter::scansideRemovalForEachScanAndSaveThem(void) {
    scansideRemovalForEachScan();
    saveCleanedScans();
    saveMapPointcloudByMergingCleanedScans();
}  // scansideRemovalForEachScanAndSaveThem

void Removerter::run(void) {
    // load scan and poses
    parseValidScanInfo();
    readValidScans();

    // construct initial map using the scans and the corresponding poses
    inno_log_info("Start to make globalmap");
    makeGlobalMap();

    // map-side removals
    for (float _rm_res : remove_resolution_list_) {
        // removeOnce(_rm_res);
        // saveCurrentStaticAndDynamicPointCloudGlobal();
        // saveCurrentStaticAndDynamicPointCloudLocal(base_node_idx_);
    }

    // // TODO
    // // map-side reverts
    // // if you want to remove as much as possible, you can omit this steps
    // for (float _rv_res : revert_resolution_list_) {
    //     revertOnce(_rv_res);
    // }

    // scan-side removals
    // scansideRemovalForEachScanAndSaveThem();
}

std::pair<int, int> resetRimgSize(const std::pair<float, float> _fov,
                                  const float _resize_ratio) {
    // default is 1 deg x 1 deg
    float alpha_vfov = _resize_ratio;
    float alpha_hfov = _resize_ratio;

    float V_FOV = _fov.first;
    float H_FOV = _fov.second;

    int NUM_RANGE_IMG_ROW = std::round(V_FOV * alpha_vfov);
    int NUM_RANGE_IMG_COL = std::round(H_FOV * alpha_hfov);

    std::pair<int, int> rimg{NUM_RANGE_IMG_ROW, NUM_RANGE_IMG_COL};
    return rimg;
}

SphericalPoint cart2sph(const PointType& _cp) {  // _cp means cartesian point
    inno_log_trace("Cartesian Point [x, y, z]: [%f,%f,%f]", _cp.x, _cp.y,
                   _cp.z);
    SphericalPoint sph_point{
        std::atan2(_cp.y, _cp.x),
        std::atan2(_cp.z, std::sqrt(_cp.x * _cp.x + _cp.y * _cp.y)),
        std::sqrt(_cp.x * _cp.x + _cp.y * _cp.y + _cp.z * _cp.z)};
    return sph_point;
}