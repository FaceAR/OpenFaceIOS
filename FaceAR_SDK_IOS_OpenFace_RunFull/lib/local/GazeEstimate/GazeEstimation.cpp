///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED AS IS?FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called open source?software licenses (Open Source
// Components?, which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensees request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltruaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltruaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltruaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltruaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>

#include "GazeEstimation.h"

cv::Point3f RaySphereIntersect(cv::Point3f rayOrigin, cv::Point3f rayDir, cv::Point3f sphereOrigin, float sphereRadius){

    float dx = rayDir.x;
    float dy = rayDir.y;
    float dz = rayDir.z;
    float x0 = rayOrigin.x;
    float y0 = rayOrigin.y;
    float z0 = rayOrigin.z;
    float cx = sphereOrigin.x;
    float cy = sphereOrigin.y;
    float cz = sphereOrigin.z;
    float r = sphereRadius;

    float a = dx*dx + dy*dy + dz*dz;
    float b = 2*dx*(x0-cx) + 2*dy*(y0-cy) + 2*dz*(z0-cz);
    float c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 + -2*(cx*x0 + cy*y0 + cz*z0) - r*r;

    float disc = b*b - 4*a*c;

    float t = (-b - sqrt(b*b - 4*a*c))/2*a;

    // This implies that the lines did not intersect, point straight ahead
    if (b*b - 4 * a*c < 0)
        return cv::Point3f(0, 0, -1);

    return rayOrigin + rayDir * t;
}

cv::Point3f GetPupilPosition(cv::Mat_<double> eyeLdmks3d){

    eyeLdmks3d = eyeLdmks3d.t();

    cv::Mat_<double> irisLdmks3d = eyeLdmks3d.rowRange(0,8);

    cv::Point3f p (mean(irisLdmks3d.col(0))[0], mean(irisLdmks3d.col(1))[0], mean(irisLdmks3d.col(2))[0]);
    return p;
}

void GazeEstimate::EstimateGaze(const LandmarkDetector::CLNF &clnf_model, cv::Point3f &gaze_absolute, float fx, float fy, float cx, float cy, bool left_eye)
{
    cv::Vec6d headPose = LandmarkDetector::GetPoseCamera(clnf_model, fx, fy, cx, cy);
    cv::Vec3d eulerAngles(headPose(3), headPose(4), headPose(5));
    cv::Matx33d rotMat = LandmarkDetector::Euler2RotationMatrix(eulerAngles);

    int part = -1;
    for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
    {
        if (left_eye && clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0)
        {
            part = i;
        }
        if (!left_eye && clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
        {
            part = i;
        }
    }

    if (part == -1)
    {
        std::cout << "Couldn't find the eye model, something wrong" << std::endl;
    }

    cv::Mat eyeLdmks3d = clnf_model.hierarchical_models[part].GetShape(fx, fy, cx, cy);

    cv::Point3f pupil = GetPupilPosition(eyeLdmks3d);
    cv::Point3f rayDir = pupil / norm(pupil);

    cv::Mat faceLdmks3d = clnf_model.GetShape(fx, fy, cx, cy);
    faceLdmks3d = faceLdmks3d.t();
    cv::Mat offset = (cv::Mat_<double>(3, 1) << 0, -3.50, 0);
    int eyeIdx = 1;
    if (left_eye)
    {
        eyeIdx = 0;
    }

    cv::Mat eyeballCentreMat = (faceLdmks3d.row(36+eyeIdx*6) + faceLdmks3d.row(39+eyeIdx*6))/2.0f + (cv::Mat(rotMat)*offset).t();

    cv::Point3f eyeballCentre = cv::Point3f(eyeballCentreMat);

    cv::Point3f gazeVecAxis = RaySphereIntersect(cv::Point3f(0,0,0), rayDir, eyeballCentre, 12) - eyeballCentre;

    gaze_absolute = gazeVecAxis / norm(gazeVecAxis);
}


void GazeEstimate::DrawGaze(cv::Mat img, const LandmarkDetector::CLNF &clnf_model, cv::Point3f gazeVecAxisLeft, cv::Point3f gazeVecAxisRight, float fx, float fy, float cx, float cy)
{
    cv::Mat cameraMat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 0);

    int part_left = -1;
    int part_right = -1;
    for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
    {
        if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0)
        {
            part_left = i;
        }
        if (clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
        {
            part_right = i;
        }
    }

    cv::Mat eyeLdmks3d_left = clnf_model.hierarchical_models[part_left].GetShape(fx, fy, cx, cy);
    cv::Point3f pupil_left = GetPupilPosition(eyeLdmks3d_left);

    cv::Mat eyeLdmks3d_right = clnf_model.hierarchical_models[part_right].GetShape(fx, fy, cx, cy);
    cv::Point3f pupil_right = GetPupilPosition(eyeLdmks3d_right);

    std::vector<cv::Point3d> points_left;
    points_left.push_back(cv::Point3d(pupil_left));
    points_left.push_back(cv::Point3d(pupil_left + gazeVecAxisLeft*50.0));

    std::vector<cv::Point3d> points_right;
    points_right.push_back(cv::Point3d(pupil_right));
    points_right.push_back(cv::Point3d(pupil_right + gazeVecAxisRight*50.0));

    cv::Mat_<double> proj_points;
    cv::Mat_<double> mesh_0 = (cv::Mat_<double>(2, 3) << points_left[0].x, points_left[0].y, points_left[0].z, points_left[1].x, points_left[1].y, points_left[1].z);
    LandmarkDetector::Project(proj_points, mesh_0, fx, fy, cx, cy);
    line(img, cv::Point(proj_points.at<double>(0,0), proj_points.at<double>(0, 1)), cv::Point(proj_points.at<double>(1, 0), proj_points.at<double>(1, 1)), cv::Scalar(110, 220, 0), 2, 8);

    cv::Mat_<double> mesh_1 = (cv::Mat_<double>(2, 3) << points_right[0].x, points_right[0].y, points_right[0].z, points_right[1].x, points_right[1].y, points_right[1].z);
    LandmarkDetector::Project(proj_points, mesh_1, fx, fy, cx, cy);
    line(img, cv::Point(proj_points.at<double>(0, 0), proj_points.at<double>(0, 1)), cv::Point(proj_points.at<double>(1, 0), proj_points.at<double>(1, 1)), cv::Scalar(110, 220, 0), 2, 8);
}
