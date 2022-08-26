#include "removert/Removerter.h"

int main(int argc, char** argv) {
    Removerter RMV;
    // pcl::PointCloud<PointType>::Ptr scan(new pcl::PointCloud<PointType>());
    // pcl::io::loadPCDFile<PointType>(
    //     "/home/xavier/repos/floam/data/lidar/1659419335178.pcd", *scan);
    // for (auto& point : scan->points) {
    //     auto temp = point.x;
    //     point.x = point.z;
    //     point.y = -point.y;
    //     point.z = temp;
    // }
    // std::pair<int, int> rimg_shape =
    //     resetRimgSize(std::pair<float, float>(25, 125), 2.5);
    // cv::Mat res =
    //     RMV.scan2RangeImg(scan, std::pair<float, float>(25, 125),
    //     rimg_shape);
    // double maxv, minv;
    // cv::minMaxLoc(res, &minv, &maxv);
    // cv::Mat scan_bit8 = (res - minv) / (maxv - minv) * 255;
    // cv::Mat bit8;
    // scan_bit8.convertTo(bit8, CV_8UC1);
    // cv::imwrite("/home/xavier/repos/removert/process.tiff", bit8);
    // std::ofstream fs;
    // fs.open("/home/xavier/repos/removert/process.csv");
    // if (fs.is_open()) {
    //     fs << cv::format(res, cv::Formatter::FormatType::FMT_CSV);
    //     fs.close();
    // } else {
    //     inno_log_error("Failed to open csv output");
    // }
    RMV.run();
    return 0;
}