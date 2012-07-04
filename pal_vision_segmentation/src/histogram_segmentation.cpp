/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2012, PAL Robotics, S.L.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PAL Robotics, S.L. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/** 
  * @file histogram_segmentation.cpp
  * @author Bence Magyar
  * @date May 2012
  * @brief Histogram based segmentation node. \
  * Using histogram backprojection, normalization, thresholding and dilate operator. \
  * Erode operator can be added by uncommenting lines \
  * in histogram_segmentation.cpp and the corresponding cfg file.
  */


#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <dynamic_reconfigure/server.h>
#include <pal_vision_segmentation/HistogramSegmentConfig.h>
#include "image_processing.h"
#include "histogram.h"

/***Variables used in callbacks***/
image_transport::Publisher mask_pub;
int threshold;
int dilate_iterations;
int dilate_size;
int erode_iterations;
int erode_size;
cv::Mat image_rect;
image_transport::Publisher image_pub;
image_transport::Publisher debug_pub;
cv::MatND target_hist;
/***end of callback section***/

void imageCb(const sensor_msgs::ImageConstPtr& msg)
{
    if(image_pub.getNumSubscribers() > 0 ||
       mask_pub.getNumSubscribers() > 0 ||
       debug_pub.getNumSubscribers() > 0)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        image_rect = cv_ptr->image;

        //double ticksBefore = cv::getTickCount();

        cv::Mat backProject;
        cv::Mat hsv;
        cv::cvtColor(image_rect, hsv, CV_BGR2HSV);
        // Quantize the hue to 30 levels
        // and the saturation to 32 levels
        // hue varies from 0 to 179, see cvtColor
        float hranges[] = { 0, 180 };
        // saturation varies from 0 (black-gray-white) to
        // 255 (pure spectrum color)
        float sranges[] = { 0, 256 };
        const float* ranges[] = { hranges, sranges };
        // we compute the histogram from the 0-th and 1-st channels
        int channels[] = {0, 1};

        cv::calcBackProject(&hsv, 1, channels, target_hist, backProject, ranges, 1, true);
        cv::normalize(backProject, backProject, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
        cv::threshold(backProject, backProject, threshold, 255, CV_THRESH_BINARY);

        cv::Mat mask, tmp1;

        if(dilate_iterations == 0 && erode_iterations == 0)
            mask = backProject;

        if(dilate_iterations > 0)
        {
            cv::dilate(backProject, erode_iterations == 0 ? mask: tmp1,
                       cv::Mat::ones(dilate_size, dilate_size, CV_8UC1),
                       cv::Point(-1, -1), dilate_iterations);
        }

        if(erode_iterations > 0)
        {
            cv::erode(dilate_iterations == 0 ? backProject : tmp1, mask,
                      cv::Mat::ones(erode_size, erode_size, CV_8UC1),
                      cv::Point(-1, -1), erode_iterations);
        }

        ros::Time now = msg->header.stamp; //ros::Time::now();

        if(mask_pub.getNumSubscribers() > 0)
        {
            cv_bridge::CvImage mask_msg;
            mask_msg.header = msg->header;
            mask_msg.header.stamp = now; //ros::Time::now();
            mask_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
            mask_msg.image = mask;
            mask_pub.publish(mask_msg.toImageMsg());
        }

        if(image_pub.getNumSubscribers() > 0)
        {
            cv::Mat masked;
            image_rect.copyTo(masked, mask);
            cv_bridge::CvImage masked_msg;
            masked_msg.header = msg->header;
            masked_msg.encoding = sensor_msgs::image_encodings::BGR8;
            masked_msg.image = masked;
            masked_msg.header.stamp  = now; //ros::Time::now();
            image_pub.publish(*masked_msg.toImageMsg());
        }

        //DEBUG
        if(debug_pub.getNumSubscribers() > 0)
        {
            cv_bridge::CvImage debug_msg;
            debug_msg.header = msg->header;
            debug_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
            debug_msg.image = backProject;
            debug_pub.publish(*debug_msg.toImageMsg());
        }

        //    ROS_INFO("imageCb runtime: %f ms",
        //             1000*(cv::getTickCount() - ticksBefore)/cv::getTickFrequency());
    }
}

void reconf_callback(pal_vision_segmentation::HistogramSegmentConfig &config, uint32_t level)
{
    threshold = config.threshold;
    dilate_iterations = config.dilate_iterations;
    dilate_size = config.dilate_size;
    erode_iterations = config.erode_iterations;
    erode_size = config.erode_size;
}

int main(int argc, char *argv[] )
{
    ros::init(argc, argv, "histogram_segmentation"); //initialize with a default name
    ros::NodeHandle nh("~"); //use node name as sufix of the namespace
    nh.param<int>("threshold", threshold, 254);
    nh.param<int>("dilate_iterations", dilate_iterations, 9);
    nh.param<int>("dilate_size", dilate_size, 7);
    nh.param<int>("erode_iterations", erode_iterations, 0);
    nh.param<int>("erode_size", erode_size, 3);

    if(argc < 2)
    {
        ROS_ERROR("Histogram segmentation needs a template to search for. Please provide it as a command line argument.");
        return -1;
    }

    std::string template_path(argv[1]);
    ROS_INFO("%s", template_path.c_str());
    cv::Mat raw_template = cv::imread(template_path);
    target_hist = pal_vision_util::calcHSVHist(raw_template);
    cv::normalize(target_hist, target_hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber rect_sub = it.subscribe("/image", 1, &imageCb);
    image_pub = it.advertise("image_masked", 1);
    mask_pub = it.advertise("mask", 1);
    debug_pub = it.advertise("debug",1);

    dynamic_reconfigure::Server<pal_vision_segmentation::HistogramSegmentConfig> server;
    dynamic_reconfigure::Server<pal_vision_segmentation::HistogramSegmentConfig>::CallbackType f;
    f = boost::bind(&reconf_callback, _1, _2);
    server.setCallback(f);

    ros::spin();
}
