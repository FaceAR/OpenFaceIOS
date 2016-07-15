///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED AS IS FOR ACADEMIC USE ONLY AND ANY EXPRESS
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
// of the Software may be covered by so-called open source software licenses (Open Source
// Components), which means any software licenses approved as open source licenses by the
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

//  Header for all external CLM/CLNF/CLM-Z methods of interest to the user
//
//
#ifndef __LANDMARK_DETECTOR_FUNC_h_
#define __LANDMARK_DETECTOR_FUNC_h_

// OpenCV includes
#include <opencv2/core/core.hpp>

#include "LandmarkDetectorParameters.h"
#include "LandmarkDetectorUtils.h"
#include "LandmarkDetectorModel.h"

//using namespace std;

namespace LandmarkDetector
{

	//================================================================================================================
	// Landmark detection in videos, need to provide an image and model parameters (default values work well)
	// Optionally can provide a bounding box from which to start tracking
	//================================================================================================================
	bool DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model, FaceModelParameters& params);
	bool DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, CLNF& clnf_model, FaceModelParameters& params);

	bool DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params);
	bool DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params);

	//================================================================================================================
	// Landmark detection in image, need to provide an image and optionally CLNF model together with parameters (default values work well)
	// Optionally can provide a bounding box in which detection is performed (this is useful if multiple faces are to be detected in images)
	//================================================================================================================
	bool DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model, FaceModelParameters& params);
	// Providing a bounding box
	bool DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params);

	//================================================
	// CLM-Z versions
	bool DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> depth_image, CLNF& clnf_model, FaceModelParameters& params);
	bool DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> depth_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params);

	//================================================================
	// Helper function for getting head pose from CLNF parameters

	// Return the current estimate of the head pose, this can be either in camera or world coordinate space
	// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
	cv::Vec6d GetPoseCamera(const CLNF& clnf_model, double fx, double fy, double cx, double cy);
	cv::Vec6d GetPoseWorld(const CLNF& clnf_model, double fx, double fy, double cx, double cy);
	
	// Getting a head pose estimate from the currently detected landmarks, with appropriate correction for perspective
	// This is because rotation estimate under orthographic assumption is only correct close to the centre of the image
	// These methods attempt to correct for that
	// The pose returned can be either in camera or world coordinates
	// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
	cv::Vec6d GetCorrectedPoseCamera(const CLNF& clnf_model, double fx, double fy, double cx, double cy);
	cv::Vec6d GetCorrectedPoseWorld(const CLNF& clnf_model, double fx, double fy, double cx, double cy);

	//===========================================================================

}
#endif

