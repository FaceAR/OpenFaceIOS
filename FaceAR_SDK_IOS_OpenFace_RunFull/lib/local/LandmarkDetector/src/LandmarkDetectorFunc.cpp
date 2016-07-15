///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED �AS IS� FOR ACADEMIC USE ONLY AND ANY EXPRESS
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
// of the Software may be covered by so-called �open source� software licenses (�Open Source
// Components�), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee�s request. Licensee will comply with the applicable terms of such licenses and to
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
//       Tadas Baltru�aitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltru�aitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "LandmarkDetectorFunc.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// System includes
#include <vector>

using namespace LandmarkDetector;

// Getting a head pose estimate from the currently detected landmarks (rotation with respect to point camera)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6d LandmarkDetector::GetPoseCamera(const CLNF& clnf_model, double fx, double fy, double cx, double cy)
{
	if(!clnf_model.detected_landmarks.empty() && clnf_model.params_global[0] != 0)
	{
		double Z = fx / clnf_model.params_global[0];
	
		double X = ((clnf_model.params_global[4] - cx) * (1.0/fx)) * Z;
		double Y = ((clnf_model.params_global[5] - cy) * (1.0/fy)) * Z;
	
		return cv::Vec6d(X, Y, Z, clnf_model.params_global[1], clnf_model.params_global[2], clnf_model.params_global[3]);
	}
	else
	{
		return cv::Vec6d(0,0,0,0,0,0);
	}
}

// Getting a head pose estimate from the currently detected landmarks (rotation in world coordinates)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6d LandmarkDetector::GetPoseWorld(const CLNF& clnf_model, double fx, double fy, double cx, double cy)
{
	if(!clnf_model.detected_landmarks.empty() && clnf_model.params_global[0] != 0)
	{
		double Z = fx / clnf_model.params_global[0];
	
		double X = ((clnf_model.params_global[4] - cx) * (1.0/fx)) * Z;
		double Y = ((clnf_model.params_global[5] - cy) * (1.0/fy)) * Z;
	
		// Here we correct for the camera orientation, for this need to determine the angle the camera makes with the head pose
		double z_x = cv::sqrt(X * X + Z * Z);
		double eul_x = atan2(Y, z_x);

		double z_y = cv::sqrt(Y * Y + Z * Z);
		double eul_y = -atan2(X, z_y);

		cv::Matx33d camera_rotation = LandmarkDetector::Euler2RotationMatrix(cv::Vec3d(eul_x, eul_y, 0));
		cv::Matx33d head_rotation = LandmarkDetector::AxisAngle2RotationMatrix(cv::Vec3d(clnf_model.params_global[1], clnf_model.params_global[2], clnf_model.params_global[3]));

		cv::Matx33d corrected_rotation = camera_rotation.t() * head_rotation;

		cv::Vec3d euler_corrected = LandmarkDetector::RotationMatrix2Euler(corrected_rotation);

		return cv::Vec6d(X, Y, Z, euler_corrected[0], euler_corrected[1], euler_corrected[2]);
	}
	else
	{
		return cv::Vec6d(0,0,0,0,0,0);
	}
}

// Getting a head pose estimate from the currently detected landmarks, with appropriate correction due to orthographic camera issue
// This is because rotation estimate under orthographic assumption is only correct close to the centre of the image
// This method returns a corrected pose estimate with respect to world coordinates (Experimental)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6d LandmarkDetector::GetCorrectedPoseWorld(const CLNF& clnf_model, double fx, double fy, double cx, double cy)
{
	if(!clnf_model.detected_landmarks.empty() && clnf_model.params_global[0] != 0)
	{
		// This is used as an initial estimate for the iterative PnP algorithm
		double Z = fx / clnf_model.params_global[0];
	
		double X = ((clnf_model.params_global[4] - cx) * (1.0/fx)) * Z;
		double Y = ((clnf_model.params_global[5] - cy) * (1.0/fy)) * Z;
 
		// Correction for orientation

		// 2D points
		cv::Mat_<double> landmarks_2D = clnf_model.detected_landmarks;

		landmarks_2D = landmarks_2D.reshape(1, 2).t();

		// 3D points
		cv::Mat_<double> landmarks_3D;
		clnf_model.pdm.CalcShape3D(landmarks_3D, clnf_model.params_local);

		landmarks_3D = landmarks_3D.reshape(1, 3).t();

		// Solving the PNP model

		// The camera matrix
		cv::Matx33d camera_matrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);
		
		cv::Vec3d vec_trans(X, Y, Z);
		cv::Vec3d vec_rot(clnf_model.params_global[1], clnf_model.params_global[2], clnf_model.params_global[3]);
		
		cv::solvePnP(landmarks_3D, landmarks_2D, camera_matrix, cv::Mat(), vec_rot, vec_trans, true);

        cv::Vec3d euler = LandmarkDetector::AxisAngle2Euler(vec_rot);
		
		return cv::Vec6d(vec_trans[0], vec_trans[1], vec_trans[2], vec_rot[0], vec_rot[1], vec_rot[2]);
	}
	else
	{
		return cv::Vec6d(0,0,0,0,0,0);
	}
}

// Getting a head pose estimate from the currently detected landmarks, with appropriate correction due to perspective projection
// This method returns a corrected pose estimate with respect to a point camera (NOTE not the world coordinates) (Experimental)
// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
cv::Vec6d LandmarkDetector::GetCorrectedPoseCamera(const CLNF& clnf_model, double fx, double fy, double cx, double cy)
{
	if(!clnf_model.detected_landmarks.empty() && clnf_model.params_global[0] != 0)
	{

		double Z = fx / clnf_model.params_global[0];
	
		double X = ((clnf_model.params_global[4] - cx) * (1.0/fx)) * Z;
		double Y = ((clnf_model.params_global[5] - cy) * (1.0/fy)) * Z;
	
		// Correction for orientation

		// 3D points
		cv::Mat_<double> landmarks_3D;
		clnf_model.pdm.CalcShape3D(landmarks_3D, clnf_model.params_local);

		landmarks_3D = landmarks_3D.reshape(1, 3).t();

		// 2D points
		cv::Mat_<double> landmarks_2D = clnf_model.detected_landmarks;
				
		landmarks_2D = landmarks_2D.reshape(1, 2).t();

		// Solving the PNP model

		// The camera matrix
		cv::Matx33d camera_matrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);
		
		cv::Vec3d vec_trans(X, Y, Z);
		cv::Vec3d vec_rot(clnf_model.params_global[1], clnf_model.params_global[2], clnf_model.params_global[3]);
		
		cv::solvePnP(landmarks_3D, landmarks_2D, camera_matrix, cv::Mat(), vec_rot, vec_trans, true);

		// Here we correct for the camera orientation, for this need to determine the angle the camera makes with the head pose
		double z_x = cv::sqrt(vec_trans[0] * vec_trans[0] + vec_trans[2] * vec_trans[2]);
		double eul_x = atan2(vec_trans[1], z_x);

		double z_y = cv::sqrt(vec_trans[1] * vec_trans[1] + vec_trans[2] * vec_trans[2]);
		double eul_y = -atan2(vec_trans[0], z_y);

		cv::Matx33d camera_rotation = LandmarkDetector::Euler2RotationMatrix(cv::Vec3d(eul_x, eul_y, 0));
		cv::Matx33d head_rotation = LandmarkDetector::AxisAngle2RotationMatrix(vec_rot);

		cv::Matx33d corrected_rotation = camera_rotation * head_rotation;

		cv::Vec3d euler_corrected = LandmarkDetector::RotationMatrix2Euler(corrected_rotation);
		
		return cv::Vec6d(vec_trans[0], vec_trans[1], vec_trans[2], euler_corrected[0], euler_corrected[1], euler_corrected[2]);
	}
	else
	{
		return cv::Vec6d(0,0,0,0,0,0);
	}
}

// If landmark detection in video succeeded create a template for use in simple tracking
void UpdateTemplate(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model)
{
	cv::Rect bounding_box;
	clnf_model.pdm.CalcBoundingBox(bounding_box, clnf_model.params_global, clnf_model.params_local);
	// Make sure the box is not out of bounds
	bounding_box = bounding_box & cv::Rect(0, 0, grayscale_image.cols, grayscale_image.rows);

	clnf_model.face_template = grayscale_image(bounding_box).clone();
}

// This method uses basic template matching in order to allow for better tracking of fast moving faces
void CorrectGlobalParametersVideo(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model, const FaceModelParameters& params)
{
	cv::Rect init_box;
	clnf_model.pdm.CalcBoundingBox(init_box, clnf_model.params_global, clnf_model.params_local);

	cv::Rect roi(init_box.x - init_box.width/2, init_box.y - init_box.height/2, init_box.width * 2, init_box.height * 2);
	roi = roi & cv::Rect(0, 0, grayscale_image.cols, grayscale_image.rows);

	int off_x = roi.x;
	int off_y = roi.y;

	double scaling = params.face_template_scale / clnf_model.params_global[0];
	cv::Mat_<uchar> image;
	if(scaling < 1)
	{
		cv::resize(clnf_model.face_template, clnf_model.face_template, cv::Size(), scaling, scaling);
		cv::resize(grayscale_image(roi), image, cv::Size(), scaling, scaling);
	}
	else
	{
		scaling = 1;
		image = grayscale_image(roi).clone();
	}
		
	// Resizing the template			
	cv::Mat corr_out;
	cv::matchTemplate(image, clnf_model.face_template, corr_out, CV_TM_CCOEFF_NORMED);

	// Actually matching it
	//double min, max;
	int max_loc[2];

	cv::minMaxIdx(corr_out, NULL, NULL, NULL, max_loc);

	cv::Rect_<double> out_bbox(max_loc[1]/scaling + off_x, max_loc[0]/scaling + off_y, clnf_model.face_template.rows / scaling, clnf_model.face_template.cols / scaling);

	double shift_x = out_bbox.x - (double)init_box.x;
	double shift_y = out_bbox.y - (double)init_box.y;
			
	clnf_model.params_global[4] = clnf_model.params_global[4] + shift_x;
	clnf_model.params_global[5] = clnf_model.params_global[5] + shift_y;
	
}

bool LandmarkDetector::DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, CLNF& clnf_model, FaceModelParameters& params)
{
	// First need to decide if the landmarks should be "detected" or "tracked"
	// Detected means running face detection and a larger search area, tracked means initialising from previous step
	// and using a smaller search area

	// Indicating that this is a first detection in video sequence or after restart
	bool initial_detection = !clnf_model.tracking_initialised;

	// Only do it if there was a face detection at all
	if(clnf_model.tracking_initialised)
	{

		// The area of interest search size will depend if the previous track was successful
		if(!clnf_model.detection_success)
		{
			params.window_sizes_current = params.window_sizes_init;
		}
		else
		{
			params.window_sizes_current = params.window_sizes_small;
		}

		// Before the expensive landmark detection step apply a quick template tracking approach
		if(params.use_face_template && !clnf_model.face_template.empty() && clnf_model.detection_success)
		{
			CorrectGlobalParametersVideo(grayscale_image, clnf_model, params);
		}

		bool track_success = clnf_model.DetectLandmarks(grayscale_image, depth_image, params);
		if(!track_success)
		{
			// Make a record that tracking failed
			clnf_model.failures_in_a_row++;
		}
		else
		{
			// indicate that tracking is a success
			clnf_model.failures_in_a_row = -1;			
			UpdateTemplate(grayscale_image, clnf_model);
		}
	}

	// This is used for both detection (if it the tracking has not been initialised yet) or if the tracking failed (however we do this every n frames, for speed)
	// This also has the effect of an attempt to reinitialise just after the tracking has failed, which is useful during large motions
	if((!clnf_model.tracking_initialised && (clnf_model.failures_in_a_row + 1) % (params.reinit_video_every * 6) == 0) 
		|| (clnf_model.tracking_initialised && !clnf_model.detection_success && params.reinit_video_every > 0 && clnf_model.failures_in_a_row % params.reinit_video_every == 0))
	{

		cv::Rect_<double> bounding_box;

		// If the face detector has not been initialised read it in
		if(clnf_model.face_detector_HAAR.empty())
		{
			clnf_model.face_detector_HAAR.load(params.face_detector_location);
			clnf_model.face_detector_location = params.face_detector_location;
		}

		cv::Point preference_det(-1, -1);
		if(clnf_model.preference_det.x != -1 && clnf_model.preference_det.y != -1)
		{
			preference_det.x = clnf_model.preference_det.x * grayscale_image.cols;
			preference_det.y = clnf_model.preference_det.y * grayscale_image.rows;
			clnf_model.preference_det = cv::Point(-1, -1);
		}

		bool face_detection_success;
//		if(params.curr_face_detector == FaceModelParameters::HOG_SVM_DETECTOR)
//		{
//			double confidence;
//			face_detection_success = LandmarkDetector::DetectSingleFaceHOG(bounding_box, grayscale_image, clnf_model.face_detector_HOG, confidence, preference_det);
//		}
//		else
        if(params.curr_face_detector == FaceModelParameters::HAAR_DETECTOR)
		{
			face_detection_success = LandmarkDetector::DetectSingleFace(bounding_box, grayscale_image, clnf_model.face_detector_HAAR, preference_det);
		}

		// Attempt to detect landmarks using the detected face (if unseccessful the detection will be ignored)
		if(face_detection_success)
		{
			// Indicate that tracking has started as a face was detected
			clnf_model.tracking_initialised = true;
						
			// Keep track of old model values so that they can be restored if redetection fails
			cv::Vec6d params_global_init = clnf_model.params_global;
			cv::Mat_<double> params_local_init = clnf_model.params_local.clone();
			double likelihood_init = clnf_model.model_likelihood;
			cv::Mat_<double> detected_landmarks_init = clnf_model.detected_landmarks.clone();
			cv::Mat_<double> landmark_likelihoods_init = clnf_model.landmark_likelihoods.clone();

			// Use the detected bounding box and empty local parameters
			clnf_model.params_local.setTo(0);
			clnf_model.pdm.CalcParams(clnf_model.params_global, bounding_box, clnf_model.params_local);		

			// Make sure the search size is large
			params.window_sizes_current = params.window_sizes_init;

			// Do the actual landmark detection (and keep it only if successful)
			bool landmark_detection_success = clnf_model.DetectLandmarks(grayscale_image, depth_image, params);

			// If landmark reinitialisation unsucessful continue from previous estimates
			// if it's initial detection however, do not care if it was successful as the validator might be wrong, so continue trackig
			// regardless
			if(!initial_detection && !landmark_detection_success)
			{

				// Restore previous estimates
				clnf_model.params_global = params_global_init;
				clnf_model.params_local = params_local_init.clone();
				clnf_model.pdm.CalcShape2D(clnf_model.detected_landmarks, clnf_model.params_local, clnf_model.params_global);
				clnf_model.model_likelihood = likelihood_init;
				clnf_model.detected_landmarks = detected_landmarks_init.clone();
				clnf_model.landmark_likelihoods = landmark_likelihoods_init.clone();

				return false;
			}
			else
			{
				clnf_model.failures_in_a_row = -1;				
				UpdateTemplate(grayscale_image, clnf_model);
				return true;
			}
		}
	}

	// if the model has not been initialised yet class it as a failure
	if(!clnf_model.tracking_initialised)
	{
		clnf_model.failures_in_a_row++;
	}

	// un-initialise the tracking
	if(	clnf_model.failures_in_a_row > 100)
	{
		clnf_model.tracking_initialised = false;
	}

	return clnf_model.detection_success;
	
}

bool LandmarkDetector::DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> &depth_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params)
{
	if(bounding_box.width > 0)
	{
		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
        clnf_model.params_local.setTo(0);
        clnf_model.pdm.CalcParams(clnf_model.params_global, bounding_box, clnf_model.params_local);

		// indicate that face was detected so initialisation is not necessary
		clnf_model.tracking_initialised = true;
	}

	return DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, params);

}

bool LandmarkDetector::DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model, FaceModelParameters& params)
{
	return DetectLandmarksInVideo(grayscale_image, cv::Mat_<float>(), clnf_model, params);
}

bool LandmarkDetector::DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params)
{
	return DetectLandmarksInVideo(grayscale_image, cv::Mat_<float>(), clnf_model, params);
}

//================================================================================================================
// Landmark detection in image, need to provide an image and optionally CLNF model together with parameters (default values work well)
// Optionally can provide a bounding box in which detection is performed (this is useful if multiple faces are to be detected in images)
//================================================================================================================

// This is the one where the actual work gets done, other DetectLandmarksInImage calls lead to this one
bool LandmarkDetector::DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> depth_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params)
{

	// Can have multiple hypotheses
    std::vector<cv::Vec3d> rotation_hypotheses;

	if(params.multi_view)
	{
		// Try out different orientation initialisations
		// It is possible to add other orientation hypotheses easilly by just pushing to this vector
		rotation_hypotheses.push_back(cv::Vec3d(0,0,0));
		rotation_hypotheses.push_back(cv::Vec3d(0,0.5236,0));
		rotation_hypotheses.push_back(cv::Vec3d(0,-0.5236,0));
		rotation_hypotheses.push_back(cv::Vec3d(0.5236,0,0));
		rotation_hypotheses.push_back(cv::Vec3d(-0.5236,0,0));
	}
	else
	{
		// Assume the face is close to frontal
		rotation_hypotheses.push_back(cv::Vec3d(0,0,0));
	}
	
	// Use the initialisation size for the landmark detection
	params.window_sizes_current = params.window_sizes_init;
	
	// Store the current best estimate
	double best_likelihood;
	cv::Vec6d best_global_parameters;
	cv::Mat_<double> best_local_parameters;
	cv::Mat_<double> best_detected_landmarks;
	cv::Mat_<double> best_landmark_likelihoods;
	bool best_success;

	// The hierarchical model parameters
    std::vector<double> best_likelihood_h(clnf_model.hierarchical_models.size());
    std::vector<cv::Vec6d> best_global_parameters_h(clnf_model.hierarchical_models.size());
    std::vector<cv::Mat_<double>> best_local_parameters_h(clnf_model.hierarchical_models.size());
    std::vector<cv::Mat_<double>> best_detected_landmarks_h(clnf_model.hierarchical_models.size());
    std::vector<cv::Mat_<double>> best_landmark_likelihoods_h(clnf_model.hierarchical_models.size());

	for(size_t hypothesis = 0; hypothesis < rotation_hypotheses.size(); ++hypothesis)
	{
		// Reset the potentially set clnf_model parameters
		clnf_model.params_local.setTo(0.0);

		for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
		{
			clnf_model.hierarchical_models[part].params_local.setTo(0.0);
		}

		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
		clnf_model.pdm.CalcParams(clnf_model.params_global, bounding_box, clnf_model.params_local, rotation_hypotheses[hypothesis]);
	
		bool success = clnf_model.DetectLandmarks(grayscale_image, depth_image, params);	

		if(hypothesis == 0 || best_likelihood < clnf_model.model_likelihood)
		{
			best_likelihood = clnf_model.model_likelihood;
			best_global_parameters = clnf_model.params_global;
			best_local_parameters = clnf_model.params_local.clone();
			best_detected_landmarks = clnf_model.detected_landmarks.clone();
			best_landmark_likelihoods = clnf_model.landmark_likelihoods.clone();
			best_success = success;
		}

		for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
		{
			if (hypothesis == 0 || best_likelihood < clnf_model.hierarchical_models[part].model_likelihood)
			{
				best_likelihood_h[part] = clnf_model.hierarchical_models[part].model_likelihood;
				best_global_parameters_h[part] = clnf_model.hierarchical_models[part].params_global;
				best_local_parameters_h[part] = clnf_model.hierarchical_models[part].params_local.clone();
				best_detected_landmarks_h[part] = clnf_model.hierarchical_models[part].detected_landmarks.clone();
				best_landmark_likelihoods_h[part] = clnf_model.hierarchical_models[part].landmark_likelihoods.clone();
			}
		}

	}

	// Store the best estimates in the clnf_model
	clnf_model.model_likelihood = best_likelihood;
	clnf_model.params_global = best_global_parameters;
	clnf_model.params_local = best_local_parameters.clone();
	clnf_model.detected_landmarks = best_detected_landmarks.clone();
	clnf_model.detection_success = best_success;
	clnf_model.landmark_likelihoods = best_landmark_likelihoods.clone();

	for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
	{
		clnf_model.hierarchical_models[part].params_global = best_global_parameters_h[part];
		clnf_model.hierarchical_models[part].params_local = best_local_parameters_h[part].clone();
		clnf_model.hierarchical_models[part].detected_landmarks = best_detected_landmarks_h[part].clone();
		clnf_model.hierarchical_models[part].landmark_likelihoods = best_landmark_likelihoods_h[part].clone();
	}

	return best_success;
}

bool LandmarkDetector::DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Mat_<float> depth_image, CLNF& clnf_model, FaceModelParameters& params)
{

	cv::Rect_<double> bounding_box;

	// If the face detector has not been initialised read it in
	if(clnf_model.face_detector_HAAR.empty())
	{
		clnf_model.face_detector_HAAR.load(params.face_detector_location);
		clnf_model.face_detector_location = params.face_detector_location;
	}
		
	// Detect the face first
//	if(params.curr_face_detector == FaceModelParameters::HOG_SVM_DETECTOR)
//	{
//		double confidence;
//		LandmarkDetector::DetectSingleFaceHOG(bounding_box, grayscale_image, clnf_model.face_detector_HOG, confidence);
//	}
//	else
    if(params.curr_face_detector == FaceModelParameters::HAAR_DETECTOR)
	{
		LandmarkDetector::DetectSingleFace(bounding_box, grayscale_image, clnf_model.face_detector_HAAR);
	}

	if(bounding_box.width == 0)
	{
		return false;
	}
	else
	{
		return DetectLandmarksInImage(grayscale_image, depth_image, bounding_box, clnf_model, params);
	}
}

// Versions not using depth images
bool LandmarkDetector::DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params)
{
	return DetectLandmarksInImage(grayscale_image, cv::Mat_<float>(), bounding_box, clnf_model, params);
}

bool LandmarkDetector::DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model, FaceModelParameters& params)
{
	return DetectLandmarksInImage(grayscale_image, cv::Mat_<float>(), clnf_model, params);
}


