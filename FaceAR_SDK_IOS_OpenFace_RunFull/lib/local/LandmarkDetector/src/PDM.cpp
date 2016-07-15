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

#include "stdafx.h"

#include "PDM.h"

// OpenCV include
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// Math includes
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;
//===========================================================================

//=============================================================================
// Orthonormalising the 3x3 rotation matrix
void Orthonormalise(cv::Matx33d &R)
{

	cv::SVD svd(R,cv::SVD::MODIFY_A);
  
	// get the orthogonal matrix from the initial rotation matrix
	cv::Mat_<double> X = svd.u*svd.vt;
  
	// This makes sure that the handedness is preserved and no reflection happened
	// by making sure the determinant is 1 and not -1
	cv::Mat_<double> W = cv::Mat_<double>::eye(3,3);
	double d = determinant(X);
	W(2,2) = determinant(X);
	cv::Mat Rt = svd.u*W*svd.vt;

	Rt.copyTo(R);

}

// A copy constructor
PDM::PDM(const PDM& other) {

	// Make sure the matrices are allocated properly
	this->mean_shape = other.mean_shape.clone();
	this->princ_comp = other.princ_comp.clone();
	this->eigen_values = other.eigen_values.clone();
}

//===========================================================================
// Clamping the parameter values to be within 3 standard deviations
void PDM::Clamp(cv::Mat_<float>& local_params, cv::Vec6d& params_global, const FaceModelParameters& parameters)
{
	double n_sigmas = 3;
	cv::MatConstIterator_<double> e_it  = this->eigen_values.begin();
	cv::MatIterator_<float> p_it =  local_params.begin();

	double v;

	// go over all parameters
	for(; p_it != local_params.end(); ++p_it, ++e_it)
	{
		// Work out the maximum value
		v = n_sigmas*sqrt(*e_it);

		// if the values is too extreme clamp it
		if(fabs(*p_it) > v)
		{
			// Dealing with positive and negative cases
			if(*p_it > 0.0)
			{
				*p_it=v;
			}
			else
			{
				*p_it=-v;
			}
		}
	}
	
	// do not let the pose get out of hand
	if(parameters.limit_pose)
	{
		if(params_global[1] > M_PI / 2)
			params_global[1] = M_PI/2;
		if(params_global[1] < -M_PI / 2)
			params_global[1] = -M_PI/2;
		if(params_global[2] > M_PI / 2)
			params_global[2] = M_PI/2;
		if(params_global[2] < -M_PI / 2)
			params_global[2] = -M_PI/2;
		if(params_global[3] > M_PI / 2)
			params_global[3] = M_PI/2;
		if(params_global[3] < -M_PI / 2)
			params_global[3] = -M_PI/2;
	}
	

}
//===========================================================================
// Compute the 3D representation of shape (in object space) using the local parameters
void PDM::CalcShape3D(cv::Mat_<double>& out_shape, const cv::Mat_<double>& p_local) const
{
	out_shape.create(mean_shape.rows, mean_shape.cols);
	out_shape = mean_shape + princ_comp*p_local;
}

//===========================================================================
// Get the 2D shape (in image space) from global and local parameters
void PDM::CalcShape2D(cv::Mat_<double>& out_shape, const cv::Mat_<double>& params_local, const cv::Vec6d& params_global) const
{

	int n = this->NumberOfPoints();

	double s = params_global[0]; // scaling factor
	double tx = params_global[4]; // x offset
	double ty = params_global[5]; // y offset

	// get the rotation matrix from the euler angles
	cv::Vec3d euler(params_global[1], params_global[2], params_global[3]);
	cv::Matx33d currRot = Euler2RotationMatrix(euler);
	
	// get the 3D shape of the object
	cv::Mat_<double> Shape_3D = mean_shape + princ_comp * params_local;

	// create the 2D shape matrix (if it has not been defined yet)
	if((out_shape.rows != mean_shape.rows) || (out_shape.cols = 1))
	{
		out_shape.create(2*n,1);
	}
	// for every vertex
	for(int i = 0; i < n; i++)
	{
		// Transform this using the weak-perspective mapping to 2D from 3D
		out_shape.at<double>(i  ,0) = s * ( currRot(0,0) * Shape_3D.at<double>(i, 0) + currRot(0,1) * Shape_3D.at<double>(i+n  ,0) + currRot(0,2) * Shape_3D.at<double>(i+n*2,0) ) + tx;
		out_shape.at<double>(i+n,0) = s * ( currRot(1,0) * Shape_3D.at<double>(i, 0) + currRot(1,1) * Shape_3D.at<double>(i+n  ,0) + currRot(1,2) * Shape_3D.at<double>(i+n*2,0) ) + ty;
	}
}

//===========================================================================
// provided the bounding box of a face and the local parameters (with optional rotation), generates the global parameters that can generate the face with the provided bounding box
// This all assumes that the bounding box describes face from left outline to right outline of the face and chin to eyebrows
void PDM::CalcParams(cv::Vec6d& out_params_global, const cv::Rect_<double>& bounding_box, const cv::Mat_<double>& params_local, const cv::Vec3d rotation)
{

	// get the shape instance based on local params
	cv::Mat_<double> current_shape(mean_shape.size());

	CalcShape3D(current_shape, params_local);

	// rotate the shape
	cv::Matx33d rotation_matrix = Euler2RotationMatrix(rotation);

	cv::Mat_<double> reshaped = current_shape.reshape(1, 3);

	cv::Mat rotated_shape = (cv::Mat(rotation_matrix) * reshaped);

	// Get the width of expected shape
	double min_x;
	double max_x;
	cv::minMaxLoc(rotated_shape.row(0), &min_x, &max_x);	

	double min_y;
	double max_y;
	cv::minMaxLoc(rotated_shape.row(1), &min_y, &max_y);

	double width = abs(min_x - max_x);
	double height = abs(min_y - max_y);

	double scaling = ((bounding_box.width / width) + (bounding_box.height / height)) / 2;

	// The estimate of face center also needs some correction
	double tx = bounding_box.x + bounding_box.width / 2;
	double ty = bounding_box.y + bounding_box.height / 2;

	// Correct it so that the bounding box is just around the minimum and maximum point in the initialised face	
	tx = tx - scaling * (min_x + max_x)/2;
    ty = ty - scaling * (min_y + max_y)/2;

	out_params_global = cv::Vec6d(scaling, rotation[0], rotation[1], rotation[2], tx, ty);
}

//===========================================================================
// provided the model parameters, compute the bounding box of a face
// The bounding box describes face from left outline to right outline of the face and chin to eyebrows
void PDM::CalcBoundingBox(cv::Rect& out_bounding_box, const cv::Vec6d& params_global, const cv::Mat_<double>& params_local)
{
	
	// get the shape instance based on local params
	cv::Mat_<double> current_shape;
	CalcShape2D(current_shape, params_local, params_global);
	
	// Get the width of expected shape
	double min_x;
	double max_x;
	cv::minMaxLoc(current_shape(cv::Rect(0, 0, 1, this->NumberOfPoints())), &min_x, &max_x);

	double min_y;
	double max_y;
	cv::minMaxLoc(current_shape(cv::Rect(0, this->NumberOfPoints(), 1, this->NumberOfPoints())), &min_y, &max_y);

	double width = abs(min_x - max_x);
	double height = abs(min_y - max_y);

	out_bounding_box = cv::Rect((int)min_x, (int)min_y, (int)width, (int)height);
}

//===========================================================================
// Calculate the PDM's Jacobian over rigid parameters (rotation, translation and scaling), the additional input W represents trust for each of the landmarks and is part of Non-Uniform RLMS 
void PDM::ComputeRigidJacobian(const cv::Mat_<float>& p_local, const cv::Vec6d& params_global, cv::Mat_<float> &Jacob, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w)
{
  	
	// number of verts
	int n = this->NumberOfPoints();
  
	Jacob.create(n * 2, 6);

	float X,Y,Z;

	float s = (float)params_global[0];
  	
	cv::Mat_<double> shape_3D_d;
	cv::Mat_<double> p_local_d;
	p_local.convertTo(p_local_d, CV_64F);
	this->CalcShape3D(shape_3D_d, p_local_d);
	
	cv::Mat_<float> shape_3D;
	shape_3D_d.convertTo(shape_3D, CV_32F);

	 // Get the rotation matrix
	cv::Vec3d euler(params_global[1], params_global[2], params_global[3]);
	cv::Matx33d currRot = Euler2RotationMatrix(euler);
	
	float r11 = (float) currRot(0,0);
	float r12 = (float) currRot(0,1);
	float r13 = (float) currRot(0,2);
	float r21 = (float) currRot(1,0);
	float r22 = (float) currRot(1,1);
	float r23 = (float) currRot(1,2);
	float r31 = (float) currRot(2,0);
	float r32 = (float) currRot(2,1);
	float r33 = (float) currRot(2,2);

	cv::MatIterator_<float> Jx = Jacob.begin();
	cv::MatIterator_<float> Jy = Jx + n * 6;

	for(int i = 0; i < n; i++)
	{
    
		X = shape_3D.at<float>(i,0);
		Y = shape_3D.at<float>(i+n,0);
		Z = shape_3D.at<float>(i+n*2,0);    
		
		// The rigid jacobian from the axis angle rotation matrix approximation using small angle assumption (R * R')
		// where R' = [1, -wz, wy
		//             wz, 1, -wx
		//             -wy, wx, 1]
		// And this is derived using the small angle assumption on the axis angle rotation matrix parametrisation

		// scaling term
		*Jx++ =  (X  * r11 + Y * r12 + Z * r13);
		*Jy++ =  (X  * r21 + Y * r22 + Z * r23);
		
		// rotation terms
		*Jx++ = (s * (Y * r13 - Z * r12) );
		*Jy++ = (s * (Y * r23 - Z * r22) );
		*Jx++ = (-s * (X * r13 - Z * r11));
		*Jy++ = (-s * (X * r23 - Z * r21));
		*Jx++ = (s * (X * r12 - Y * r11) );
		*Jy++ = (s * (X * r22 - Y * r21) );
		
		// translation terms
		*Jx++ = 1.0f;
		*Jy++ = 0.0f;
		*Jx++ = 0.0f;
		*Jy++ = 1.0f;

	}

	cv::Mat Jacob_w = cv::Mat::zeros(Jacob.rows, Jacob.cols, Jacob.type());
	
	Jx =  Jacob.begin();
	Jy =  Jx + n*6;

	cv::MatIterator_<float> Jx_w =  Jacob_w.begin<float>();
	cv::MatIterator_<float> Jy_w =  Jx_w + n*6;

	// Iterate over all Jacobian values and multiply them by the weight in diagonal of W
	for(int i = 0; i < n; i++)
	{
		float w_x = W.at<float>(i, i);
		float w_y = W.at<float>(i+n, i+n);

		for(int j = 0; j < Jacob.cols; ++j)
		{
			*Jx_w++ = *Jx++ * w_x;
			*Jy_w++ = *Jy++ * w_y;
		}		
	}

	Jacob_t_w = Jacob_w.t();
}

//===========================================================================
// Calculate the PDM's Jacobian over all parameters (rigid and non-rigid), the additional input W represents trust for each of the landmarks and is part of Non-Uniform RLMS
void PDM::ComputeJacobian(const cv::Mat_<float>& params_local, const cv::Vec6d& params_global, cv::Mat_<float> &Jacobian, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w)
{ 
	
	// number of vertices
	int n = this->NumberOfPoints();
		
	// number of non-rigid parameters
	int m = this->NumberOfModes();

	Jacobian.create(n * 2, 6 + m);
	
	float X,Y,Z;
	
	float s = (float) params_global[0];
  	
	cv::Mat_<double> shape_3D_d;
	cv::Mat_<double> p_local_d;
	params_local.convertTo(p_local_d, CV_64F);
	this->CalcShape3D(shape_3D_d, p_local_d);
	
	cv::Mat_<float> shape_3D;
	shape_3D_d.convertTo(shape_3D, CV_32F);

	cv::Vec3d euler(params_global[1], params_global[2], params_global[3]);
	cv::Matx33d currRot = Euler2RotationMatrix(euler);
	
	float r11 = (float) currRot(0,0);
	float r12 = (float) currRot(0,1);
	float r13 = (float) currRot(0,2);
	float r21 = (float) currRot(1,0);
	float r22 = (float) currRot(1,1);
	float r23 = (float) currRot(1,2);
	float r31 = (float) currRot(2,0);
	float r32 = (float) currRot(2,1);
	float r33 = (float) currRot(2,2);

	cv::MatIterator_<float> Jx =  Jacobian.begin();
	cv::MatIterator_<float> Jy =  Jx + n * (6 + m);
	cv::MatConstIterator_<double> Vx =  this->princ_comp.begin();
	cv::MatConstIterator_<double> Vy =  Vx + n*m;
	cv::MatConstIterator_<double> Vz =  Vy + n*m;

	for(int i = 0; i < n; i++)
	{
    
		X = shape_3D.at<float>(i,0);
		Y = shape_3D.at<float>(i+n,0);
		Z = shape_3D.at<float>(i+n*2,0);    
    
		// The rigid jacobian from the axis angle rotation matrix approximation using small angle assumption (R * R')
		// where R' = [1, -wz, wy
		//             wz, 1, -wx
		//             -wy, wx, 1]
		// And this is derived using the small angle assumption on the axis angle rotation matrix parametrisation

		// scaling term
		*Jx++ = (X  * r11 + Y * r12 + Z * r13);
		*Jy++ = (X  * r21 + Y * r22 + Z * r23);
		
		// rotation terms
		*Jx++ = (s * (Y * r13 - Z * r12) );
		*Jy++ = (s * (Y * r23 - Z * r22) );
		*Jx++ = (-s * (X * r13 - Z * r11));
		*Jy++ = (-s * (X * r23 - Z * r21));
		*Jx++ = (s * (X * r12 - Y * r11) );
		*Jy++ = (s * (X * r22 - Y * r21) );
		
		// translation terms
		*Jx++ = 1.0f;
		*Jy++ = 0.0f;
		*Jx++ = 0.0f;
		*Jy++ = 1.0f;

		for(int j = 0; j < m; j++,++Vx,++Vy,++Vz)
		{
			// How much the change of the non-rigid parameters (when object is rotated) affect 2D motion
			*Jx++ = (float) ( s*(r11*(*Vx) + r12*(*Vy) + r13*(*Vz)) );
			*Jy++ = (float) ( s*(r21*(*Vx) + r22*(*Vy) + r23*(*Vz)) );
		}
	}	

	// Adding the weights here
	cv::Mat Jacob_w = Jacobian.clone();
	
	if(cv::trace(W)[0] != W.rows) 
	{
		Jx =  Jacobian.begin();
		Jy =  Jx + n*(6+m);

		cv::MatIterator_<float> Jx_w =  Jacob_w.begin<float>();
		cv::MatIterator_<float> Jy_w =  Jx_w + n*(6+m);

		// Iterate over all Jacobian values and multiply them by the weight in diagonal of W
		for(int i = 0; i < n; i++)
		{
			float w_x = W.at<float>(i, i);
			float w_y = W.at<float>(i+n, i+n);

			for(int j = 0; j < Jacobian.cols; ++j)
			{
				*Jx_w++ = *Jx++ * w_x;
				*Jy_w++ = *Jy++ * w_y;
			}
		}
	}
	Jacob_t_w = Jacob_w.t();

}

//===========================================================================
// Updating the parameters (more details in my thesis)
void PDM::UpdateModelParameters(const cv::Mat_<float>& delta_p, cv::Mat_<float>& params_local, cv::Vec6d& params_global)
{

	// The scaling and translation parameters can be just added
	params_global[0] += (double)delta_p.at<float>(0,0);
	params_global[4] += (double)delta_p.at<float>(4,0);
	params_global[5] += (double)delta_p.at<float>(5,0);

	// get the original rotation matrix	
	cv::Vec3d eulerGlobal(params_global[1], params_global[2], params_global[3]);
	cv::Matx33d R1 = Euler2RotationMatrix(eulerGlobal);

	// construct R' = [1, -wz, wy
	//               wz, 1, -wx
	//               -wy, wx, 1]
	cv::Matx33d R2 = cv::Matx33d::eye();

	R2(1,2) = -1.0*(R2(2,1) = (double)delta_p.at<float>(1,0));
	R2(2,0) = -1.0*(R2(0,2) = (double)delta_p.at<float>(2,0));
	R2(0,1) = -1.0*(R2(1,0) = (double)delta_p.at<float>(3,0));
	
	// Make sure it's orthonormal
	Orthonormalise(R2);

	// Combine rotations
	cv::Matx33d R3 = R1 *R2;

	// Extract euler angle (through axis angle first to make sure it's legal)
	cv::Vec3d axis_angle = RotationMatrix2AxisAngle(R3);
	cv::Vec3d euler = AxisAngle2Euler(axis_angle);

	params_global[1] = euler[0];
	params_global[2] = euler[1];
	params_global[3] = euler[2];

	// Local parameter update, just simple addition
	if(delta_p.rows > 6)
	{
		params_local = params_local + delta_p(cv::Rect(0,6,1, this->NumberOfModes()));
	}

}

void PDM::CalcParams(cv::Vec6d& out_params_global, const cv::Mat_<double>& out_params_local, const cv::Mat_<double>& landmark_locations, const cv::Vec3d rotation)
{
		
	int m = this->NumberOfModes();
	int n = this->NumberOfPoints();

	cv::Mat_<int> visi_ind_2D(n * 2, 1, 1);
	cv::Mat_<int> visi_ind_3D(3 * n , 1, 1);

	for(int i = 0; i < n; ++i)
	{
		// If the landmark is invisible indicate this
		if(landmark_locations.at<double>(i) == 0)
		{
			visi_ind_2D.at<int>(i) = 0;
			visi_ind_2D.at<int>(i+n) = 0;
			visi_ind_3D.at<int>(i) = 0;
			visi_ind_3D.at<int>(i+n) = 0;
			visi_ind_3D.at<int>(i+2*n) = 0;
		}
	}

	// As this might be subsampled have special versions
	cv::Mat_<double> M(0, mean_shape.cols, 0.0);
	cv::Mat_<double> V(0, princ_comp.cols, 0.0);

	for(int i = 0; i < n * 3; ++i)
	{
		if(visi_ind_3D.at<int>(i) == 1)
		{
			cv::vconcat(M, this->mean_shape.row(i), M);
			cv::vconcat(V, this->princ_comp.row(i), V);
		}
	}

	cv::Mat_<double> m_old = this->mean_shape.clone();
	cv::Mat_<double> v_old = this->princ_comp.clone();
//    std::cout << "m_old = " << m_old.t() << std::endl;
//    std::cout << "v_old = " << v_old << std::endl;

	this->mean_shape = M;
	this->princ_comp = V;

	// The new number of points
	n  = M.rows / 3;

	// Extract the relevant landmark locations
	cv::Mat_<double> landmark_locs_vis(n*2, 1, 0.0);
	int k = 0;
	for(int i = 0; i < visi_ind_2D.rows; ++i)
	{
		if(visi_ind_2D.at<int>(i) == 1)
		{
			landmark_locs_vis.at<double>(k) = landmark_locations.at<double>(i);
			k++;
		}		
	}

	// Compute the initial global parameters
	double min_x;
	double max_x;
	cv::minMaxLoc(landmark_locations(cv::Rect(0, 0, 1, this->NumberOfPoints())), &min_x, &max_x);

	double min_y;
	double max_y;
	cv::minMaxLoc(landmark_locations(cv::Rect(0, this->NumberOfPoints(), 1, this->NumberOfPoints())), &min_y, &max_y);

	double width = std::abs(min_x - max_x);
    double height = std::abs(min_y - max_y);

	cv::Rect model_bbox;
	CalcBoundingBox(model_bbox, cv::Vec6d(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), cv::Mat_<double>(this->NumberOfModes(), 1, 0.0));

	cv::Rect bbox((int)min_x, (int)min_y, (int)width, (int)height);

	double scaling = ((width / model_bbox.width) + (height / model_bbox.height)) / 2;
        
	cv::Vec3d rotation_init = rotation;
	cv::Matx33d R = Euler2RotationMatrix(rotation_init);
	cv::Vec2d translation((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
    
	cv::Mat_<float> loc_params(this->NumberOfModes(),1, 0.0);
	cv::Vec6d glob_params(scaling, rotation_init[0], rotation_init[1], rotation_init[2], translation[0], translation[1]);

	// get the 3D shape of the object
	cv::Mat_<double> loc_params_d;
	loc_params.convertTo(loc_params_d, CV_64F);
	cv::Mat_<double> shape_3D = M + V * loc_params_d;

	cv::Mat_<double> curr_shape(2*n, 1);
	
	// for every vertex
	for(int i = 0; i < n; i++)
	{
		// Transform this using the weak-perspective mapping to 2D from 3D
		curr_shape.at<double>(i  ,0) = scaling * ( R(0,0) * shape_3D.at<double>(i, 0) + R(0,1) * shape_3D.at<double>(i+n  ,0) + R(0,2) * shape_3D.at<double>(i+n*2,0) ) + translation[0];
		curr_shape.at<double>(i+n,0) = scaling * ( R(1,0) * shape_3D.at<double>(i, 0) + R(1,1) * shape_3D.at<double>(i+n  ,0) + R(1,2) * shape_3D.at<double>(i+n*2,0) ) + translation[1];
	}
		    
    double currError = cv::norm(curr_shape - landmark_locs_vis);

	cv::Mat_<float> regularisations = cv::Mat_<double>::zeros(1, 6 + m);

	double reg_factor = 1;

	// Setting the regularisation to the inverse of eigenvalues
	cv::Mat(reg_factor / this->eigen_values).copyTo(regularisations(cv::Rect(6, 0, m, 1)));
	cv::Mat_<double> regTerm_d = cv::Mat::diag(regularisations.t());
	regTerm_d.convertTo(regularisations, CV_32F);    
    
	cv::Mat_<float> WeightMatrix = cv::Mat_<float>::eye(n*2, n*2);

	int not_improved_in = 0;

    for (size_t i = 0; i < 1000; ++i)
	{
		// get the 3D shape of the object
		cv::Mat_<double> loc_params_d;
		loc_params.convertTo(loc_params_d, CV_64F);
		shape_3D = M + V * loc_params_d;

		shape_3D = shape_3D.reshape(1, 3);

		cv::Matx23d R_2D(R(0,0), R(0,1), R(0,2), R(1,0), R(1,1), R(1,2));

		cv::Mat_<double> curr_shape_2D = scaling * shape_3D.t() * cv::Mat(R_2D).t();
        curr_shape_2D.col(0) = curr_shape_2D.col(0) + translation(0);
		curr_shape_2D.col(1) = curr_shape_2D.col(1) + translation(1);

		curr_shape_2D = cv::Mat(curr_shape_2D.t()).reshape(1, n * 2);
		
		cv::Mat_<float> error_resid;
		cv::Mat(landmark_locs_vis - curr_shape_2D).convertTo(error_resid, CV_32F);
        
		cv::Mat_<float> J, J_w_t;
		this->ComputeJacobian(loc_params, glob_params, J, WeightMatrix, J_w_t);
        
		// projection of the meanshifts onto the jacobians (using the weighted Jacobian, see Baltrusaitis 2013)
		cv::Mat_<float> J_w_t_m = J_w_t * error_resid;

		// Add the regularisation term
		J_w_t_m(cv::Rect(0,6,1, m)) = J_w_t_m(cv::Rect(0,6,1, m)) - regularisations(cv::Rect(6,6, m, m)) * loc_params;

		cv::Mat_<float> Hessian = J_w_t * J;

		// Add the Tikhonov regularisation
		Hessian = Hessian + regularisations;

		// Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
		cv::Mat_<float> param_update;
		cv::solve(Hessian, J_w_t_m, param_update, CV_CHOLESKY);

		// To not overshoot, have the gradient decent rate a bit smaller
		param_update = 0.5 * param_update;

		UpdateModelParameters(param_update, loc_params, glob_params);		
        
        scaling = glob_params[0];
		rotation_init[0] = glob_params[1];
		rotation_init[1] = glob_params[2];
		rotation_init[2] = glob_params[3];

		translation[0] = glob_params[4];
		translation[1] = glob_params[5];
        
		R = Euler2RotationMatrix(rotation_init);

		R_2D(0,0) = R(0,0);R_2D(0,1) = R(0,1); R_2D(0,2) = R(0,2);
		R_2D(1,0) = R(1,0);R_2D(1,1) = R(1,1); R_2D(1,2) = R(1,2); 

		curr_shape_2D = scaling * shape_3D.t() * cv::Mat(R_2D).t();
        curr_shape_2D.col(0) = curr_shape_2D.col(0) + translation(0);
		curr_shape_2D.col(1) = curr_shape_2D.col(1) + translation(1);

		curr_shape_2D = cv::Mat(curr_shape_2D.t()).reshape(1, n * 2);
        
        double error = cv::norm(curr_shape_2D - landmark_locs_vis);
        
        if(0.999 * currError < error)
		{
			not_improved_in++;
			if (not_improved_in == 5)
			{
	            break;
			}
		}

		currError = error;
        
	}

	out_params_global = glob_params;
	loc_params.convertTo(out_params_local, CV_64F);
    	
	this->mean_shape = m_old;
	this->princ_comp = v_old;


}

void PDM::Read(std::string location)
{
  	
    std::ifstream pdmLoc(location, std::ios_base::in);

	LandmarkDetector::SkipComments(pdmLoc);

	// Reading mean values
	LandmarkDetector::ReadMat(pdmLoc,mean_shape);
	
	LandmarkDetector::SkipComments(pdmLoc);

	// Reading principal components
	LandmarkDetector::ReadMat(pdmLoc,princ_comp);
	
	LandmarkDetector::SkipComments(pdmLoc);
	
	// Reading eigenvalues	
	LandmarkDetector::ReadMat(pdmLoc,eigen_values);

}

