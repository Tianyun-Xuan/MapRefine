#include "ground_segmentation.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <list>
#include <memory>
#include <thread>

#include "../innolog/logger.h"

using namespace std::chrono_literals;

GroundSegmentation::GroundSegmentation(const GroundSegmentationParams &params)
    : params_(params),
      segments_(params.n_segments,
                Segment(params.n_bins, params.min_slope, params.max_slope,
                        params.max_error_square, params.long_threshold,
                        params.max_long_height, params.max_start_height,
                        params.sensor_height)) {}

void GroundSegmentation::segment(const CloudPtr &cloud, CloudPtr &ground,
                                 CloudPtr &no_ground) {
    std::vector<int> segmentation;
    segmentation.resize(cloud->size(), 0);
    bin_index_.resize(cloud->size());
    segment_coordinates_.resize(cloud->size());
    resetSegments();
    insertPoints(*cloud);
    std::list<PointLine> lines;
    getLines(NULL);
    assignCluster(&segmentation);
    for (size_t i = 0; i < cloud->size(); ++i) {
        pcl::PointXYZI tt(cloud->points[i].x, cloud->points[i].y,
                          cloud->points[i].z);
        if (segmentation[i] == 1)
            ground->emplace_back(cloud->points[i]);
        else
            no_ground->emplace_back(cloud->points[i]);
    }
    inno_log_info("ground_points size %zu", ground->size());
    inno_log_info("no_ground_points size %zu", no_ground->size());
}

void GroundSegmentation::getLines(std::list<PointLine> *lines) {
    std::mutex line_mutex;
    std::vector<std::thread> thread_vec(params_.n_threads);
    unsigned int i;
    for (i = 0; i < params_.n_threads; ++i) {
        const unsigned int start_index =
            params_.n_segments / params_.n_threads * i;
        const unsigned int end_index =
            params_.n_segments / params_.n_threads * (i + 1);
        thread_vec[i] = std::thread(&GroundSegmentation::lineFitThread, this,
                                    start_index, end_index, lines, &line_mutex);
    }
    for (auto it = thread_vec.begin(); it != thread_vec.end(); ++it) {
        it->join();
    }
}

void GroundSegmentation::lineFitThread(const unsigned int start_index,
                                       const unsigned int end_index,
                                       std::list<PointLine> *lines,
                                       std::mutex *lines_mutex) {
    const bool visualize = lines;
    // const double seg_step = 2 * M_PI / params_.n_segments;
    double angle = -params_.h_fov / 2 + params_.seg_step / 2 +
                   params_.seg_step * start_index;
    for (unsigned int i = start_index; i < end_index; ++i) {
        segments_[i].fitSegmentLines();
        // Convert lines to 3d if we want to.
        if (visualize) {
            std::list<Segment::Line> segment_lines;
            segments_[i].getLines(&segment_lines);
            for (auto line_iter = segment_lines.begin();
                 line_iter != segment_lines.end(); ++line_iter) {
                const PointType start = minZPointTo3d(line_iter->first, angle);
                const PointType end = minZPointTo3d(line_iter->second, angle);
                lines_mutex->lock();
                lines->emplace_back(start, end);
                lines_mutex->unlock();
            }

            angle += params_.seg_step;
        }
    }
}

void GroundSegmentation::getMinZPointCloud(PointCloud *cloud) {
    cloud->reserve(params_.n_segments * params_.n_bins);
    // const double seg_step = 2 * M_PI / params_.n_segments;
    double angle = -params_.h_fov / 2 + params_.seg_step / 2;
    for (auto seg_iter = segments_.begin(); seg_iter != segments_.end();
         ++seg_iter) {
        for (auto bin_iter = seg_iter->begin(); bin_iter != seg_iter->end();
             ++bin_iter) {
            const PointType min =
                minZPointTo3d(bin_iter->getMinZPoint(), angle);
            cloud->push_back(min);
        }

        angle += params_.seg_step;
    }
}

void GroundSegmentation::resetSegments() {
    segments_ = std::vector<Segment>(
        params_.n_segments,
        Segment(params_.n_bins, params_.min_slope, params_.max_slope,
                params_.max_error_square, params_.long_threshold,
                params_.max_long_height, params_.max_start_height,
                params_.sensor_height));
}

PointType GroundSegmentation::minZPointTo3d(const Bin::MinZPoint &min_z_point,
                                            const double &angle) {
    PointType point;
    point.x = cos(angle) * min_z_point.d;
    point.y = sin(angle) * min_z_point.d;
    point.z = min_z_point.z;
    return point;
}

void GroundSegmentation::assignCluster(std::vector<int> *segmentation) {
    std::vector<std::thread> thread_vec(params_.n_threads);
    const size_t cloud_size = segmentation->size();
    for (unsigned int i = 0; i < params_.n_threads; ++i) {
        const unsigned int start_index = cloud_size / params_.n_threads * i;
        const unsigned int end_index = cloud_size / params_.n_threads * (i + 1);
        thread_vec[i] = std::thread(&GroundSegmentation::assignClusterThread,
                                    this, start_index, end_index, segmentation);
    }
    for (auto it = thread_vec.begin(); it != thread_vec.end(); ++it) {
        it->join();
    }
}

void GroundSegmentation::assignClusterThread(const unsigned int &start_index,
                                             const unsigned int &end_index,
                                             std::vector<int> *segmentation) {
    // const double segment_step = 2 * M_PI / params_.n_segments;
    for (unsigned int i = start_index; i < end_index; ++i) {
        // current Bin lowest point
        Bin::MinZPoint point_2d = segment_coordinates_[i];
        const int segment_index = bin_index_[i].first;
        if (segment_index >= 0) {
            double dist = segments_[segment_index].verticalDistanceToLine(
                point_2d.d, point_2d.z);
            // Search neighboring segments.
            int steps = 1;

            while (dist == -1 &&
                   steps * params_.seg_step < params_.line_search_angle) {
                // Fix indices that are out of bounds.
                int index_1 = segment_index + steps;
                while (index_1 >= params_.n_segments)
                    index_1 -= params_.n_segments;
                int index_2 = segment_index - steps;
                while (index_2 < 0) index_2 += params_.n_segments;
                // Get distance to neighboring lines.
                const double dist_1 = segments_[index_1].verticalDistanceToLine(
                    point_2d.d, point_2d.z);
                const double dist_2 = segments_[index_2].verticalDistanceToLine(
                    point_2d.d, point_2d.z);
                // (Xavier: This make me confusion, why select a larger
                // distance?) Select larger distance if both segments return a
                // valid distance.
                // if (dist_1 < dist && dist_1 != -1) {
                //     dist = dist_1;
                // }
                // if (dist_2 < dist && dist_2 != -1) {
                //     dist = dist_2;
                // }
                // Target is to pick out the minimum non -1 value
                dist = std::min(dist_1, dist_2) < 0.f ? std::max(dist_1, dist_2)
                                                      : dist;
                ++steps;
            }
            if (dist < params_.max_dist_to_line && dist != -1) {
                segmentation->at(i) = 1;
            }
        }
    }
}

void GroundSegmentation::getMinZPoints(PointCloud *out_cloud) {
    // const double seg_step = 2 * M_PI / params_.n_segments;
    const double bin_step =
        (sqrt(params_.r_max_square) - sqrt(params_.r_min_square)) /
        params_.n_bins;
    const double r_min = sqrt(params_.r_min_square);
    double angle = -params_.h_fov / 2 + params_.seg_step / 2;
    for (auto seg_iter = segments_.begin(); seg_iter != segments_.end();
         ++seg_iter) {
        double dist = r_min + bin_step / 2;
        for (auto bin_iter = seg_iter->begin(); bin_iter != seg_iter->end();
             ++bin_iter) {
            PointType point;
            if (bin_iter->hasPoint()) {
                Bin::MinZPoint min_z_point(bin_iter->getMinZPoint());
                point.x = cos(angle) * min_z_point.d;
                point.y = sin(angle) * min_z_point.d;
                point.z = min_z_point.z;

                out_cloud->push_back(point);
            }
            dist += bin_step;
        }
        angle += params_.seg_step;
    }
}

void GroundSegmentation::insertPoints(const PointCloud &cloud) {
    // (Xavier: no lock for container, may not so safe)
    std::vector<std::thread> threads(params_.n_threads);
    const size_t points_per_thread = cloud.size() / params_.n_threads;
    // Launch threads.
    for (unsigned int i = 0; i < params_.n_threads - 1; ++i) {
        const size_t start_index = i * points_per_thread;
        const size_t end_index = (i + 1) * points_per_thread;
        threads[i] = std::thread(&GroundSegmentation::insertionThread, this,
                                 cloud, start_index, end_index);
    }
    // Launch last thread which might have more points than others.
    const size_t start_index = (params_.n_threads - 1) * points_per_thread;
    const size_t end_index = cloud.size();
    threads[params_.n_threads - 1] =
        std::thread(&GroundSegmentation::insertionThread, this, cloud,
                    start_index, end_index);
    // Wait for threads to finish.
    for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
    }
    // insertionThread(cloud, 0, cloud.size());
}

void GroundSegmentation::insertionThread(const PointCloud &cloud,
                                         const size_t start_index,
                                         const size_t end_index) {
    // const double segment_step = 2 * M_PI / params_.n_segments;
    const double bin_step =
        (sqrt(params_.r_max_square) - sqrt(params_.r_min_square)) /
        params_.n_bins;
    const double r_min = sqrt(params_.r_min_square);
    for (unsigned int i = start_index; i < end_index; ++i) {
        PointType point(cloud[i]);
        const double range_square = point.x * point.x + point.y * point.y;
        const double range = sqrt(range_square);
        if (range_square < params_.r_max_square &&
            range_square > params_.r_min_square) {
            const double angle = std::atan2(point.y, point.x);
            const unsigned int bin_index = (range - r_min) / bin_step;
            const unsigned int segment_index =
                (angle + params_.h_fov / 2) / params_.seg_step;
            const unsigned int segment_index_clamped =
                segment_index == params_.n_segments ? 0 : segment_index;
            segments_[segment_index_clamped][bin_index].addPoint(range,
                                                                 point.z);
            bin_index_[i] = std::make_pair(segment_index_clamped, bin_index);
        } else {
            bin_index_[i] = std::make_pair<int, int>(-1, -1);
        }
        segment_coordinates_[i] = Bin::MinZPoint(range, point.z);
    }
}