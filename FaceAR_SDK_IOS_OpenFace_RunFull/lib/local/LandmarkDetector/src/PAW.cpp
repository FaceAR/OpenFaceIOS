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

#include "PAW.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;

// Copy constructor
PAW::PAW(const PAW& other) : destination_landmarks(other.destination_landmarks.clone()), source_landmarks(other.source_landmarks.clone()), triangulation(other.triangulation.clone()),
triangle_id(other.triangle_id.clone()), pixel_mask(other.pixel_mask.clone()), coefficients(other.coefficients.clone()), alpha(other.alpha.clone()), beta(other.beta.clone()), map_x(other.map_x.clone()), map_y(other.map_y.clone())
{
	this->number_of_pixels = other.number_of_pixels;
	this->min_x = other.min_x;
	this->min_y = other.min_y;
}

// A constructor from destination shape and triangulation
PAW::PAW(const cv::Mat_<double>& destination_shape, const cv::Mat_<int>& triangulation)
{
	// Initialise some variables directly
	this->destination_landmarks = destination_shape;
	this->triangulation = triangulation;

	int num_points = destination_shape.rows/2;

	int num_tris = triangulation.rows;
	
	// Pre-compute the rest
    alpha = cv::Mat_<double>(num_tris, 3);
    beta = cv::Mat_<double>(num_tris, 3);
    
	cv::Mat_<double> xs = destination_shape(cv::Rect(0, 0, 1, num_points));
	cv::Mat_<double> ys = destination_shape(cv::Rect(0, num_points, 1, num_points));
    
	// Create a vector representation of the control points
    std::vector<std::vector<double>> destination_points;

	for (int tri = 0; tri < num_tris; ++tri)
	{	
		int j = triangulation.at<int>(tri, 0);
		int k = triangulation.at<int>(tri, 1);
		int l = triangulation.at<int>(tri, 2);

        double c1 = ys.at<double>(l) - ys.at<double>(j);
        double c2 = xs.at<double>(l) - xs.at<double>(j);
        double c4 = ys.at<double>(k) - ys.at<double>(j);
        double c3 = xs.at<double>(k) - xs.at<double>(j);
        		
        double c5 = c3*c1 - c2*c4;

        alpha.at<double>(tri, 0) = (ys.at<double>(j) * c2 - xs.at<double>(j) * c1) / c5;
        alpha.at<double>(tri, 1) = c1/c5;
        alpha.at<double>(tri, 2) = -c2/c5;

        beta.at<double>(tri, 0) = (xs.at<double>(j) * c4 - ys.at<double>(j) * c3)/c5;
        beta.at<double>(tri, 1) = -c4/c5;
        beta.at<double>(tri, 2) = c3/c5;

		// Add points corresponding to triangles as optimisation
        std::vector<double> triangle_points(10);

		triangle_points[0] = xs.at<double>(j);
		triangle_points[1] = ys.at<double>(j);
		triangle_points[2] = xs.at<double>(k);
		triangle_points[3] = ys.at<double>(k);
		triangle_points[4] = xs.at<double>(l);
		triangle_points[5] = ys.at<double>(l);
		
		cv::Vec3d xs_three(triangle_points[0], triangle_points[2], triangle_points[4]);
		cv::Vec3d ys_three(triangle_points[1], triangle_points[3], triangle_points[5]);

		double min_x, max_x, min_y, max_y;
		cv::minMaxIdx(xs_three, &min_x, &max_x);
		cv::minMaxIdx(ys_three, &min_y, &max_y);

		triangle_points[6] = max_x;
		triangle_points[7] = max_y;

		triangle_points[8] = min_x;
		triangle_points[9] = min_y;

		destination_points.push_back(triangle_points);

	}

	double max_x;
	double max_y;

	minMaxLoc(xs, &min_x, &max_x);
	minMaxLoc(ys, &min_y, &max_y);

	int w = (int)(max_x - min_x + 1.5);
    int h = (int)(max_y - min_y + 1.5);
    
	// Round the min_x and min_y for simplicity?

    pixel_mask = cv::Mat_<uchar>(h, w, (uchar)0);
    triangle_id = cv::Mat_<int>(h, w, -1);
        
	int curr_tri = -1;

	for(int y = 0; y < pixel_mask.rows; y++)
	{
		for(int x = 0; x < pixel_mask.cols; x++)
		{
			curr_tri = findTriangle(cv::Point_<double>(x + min_x, y + min_y), destination_points, curr_tri);
			// If there is a triangle at this location
            if(curr_tri != -1)
			{
				triangle_id.at<int>(y, x) = curr_tri;
                pixel_mask.at<uchar>(y, x) = 1;
			}	
		}
	}
    	
	// Preallocate maps and coefficients
	coefficients.create(num_tris, 6);
	map_x.create(pixel_mask.rows,pixel_mask.cols);
	map_y.create(pixel_mask.rows,pixel_mask.cols);


}

// Manually define min and max values
PAW::PAW(const cv::Mat_<double>& destination_shape, const cv::Mat_<int>& triangulation, double in_min_x, double in_min_y, double in_max_x, double in_max_y)
{
	// Initialise some variables directly
	this->destination_landmarks = destination_shape;
	this->triangulation = triangulation;

	int num_points = destination_shape.rows/2;

	int num_tris = triangulation.rows;
	
	// Pre-compute the rest
    alpha = cv::Mat_<double>(num_tris, 3);
    beta = cv::Mat_<double>(num_tris, 3);
    
	cv::Mat_<double> xs = destination_shape(cv::Rect(0, 0, 1, num_points));
	cv::Mat_<double> ys = destination_shape(cv::Rect(0, num_points, 1, num_points));

	// Create a vector representation of the control points
    std::vector<std::vector<double>> destination_points;
    
	for (int tri = 0; tri < num_tris; ++tri)
	{	
		int j = triangulation.at<int>(tri, 0);
		int k = triangulation.at<int>(tri, 1);
		int l = triangulation.at<int>(tri, 2);

        double c1 = ys.at<double>(l) - ys.at<double>(j);
        double c2 = xs.at<double>(l) - xs.at<double>(j);
        double c4 = ys.at<double>(k) - ys.at<double>(j);
        double c3 = xs.at<double>(k) - xs.at<double>(j);
        		
        double c5 = c3*c1 - c2*c4;

        alpha.at<double>(tri, 0) = (ys.at<double>(j) * c2 - xs.at<double>(j) * c1) / c5;
        alpha.at<double>(tri, 1) = c1/c5;
        alpha.at<double>(tri, 2) = -c2/c5;

        beta.at<double>(tri, 0) = (xs.at<double>(j) * c4 - ys.at<double>(j) * c3)/c5;
        beta.at<double>(tri, 1) = -c4/c5;
        beta.at<double>(tri, 2) = c3/c5;

		// Add points corresponding to triangles as optimisation
        std::vector<double> triangle_points(10);

		triangle_points[0] = xs.at<double>(j);
		triangle_points[1] = ys.at<double>(j);
		triangle_points[2] = xs.at<double>(k);
		triangle_points[3] = ys.at<double>(k);
		triangle_points[4] = xs.at<double>(l);
		triangle_points[5] = ys.at<double>(l);
		
		cv::Vec3d xs_three(triangle_points[0], triangle_points[2], triangle_points[4]);
		cv::Vec3d ys_three(triangle_points[1], triangle_points[3], triangle_points[5]);

		double min_x, max_x, min_y, max_y;
		cv::minMaxIdx(xs_three, &min_x, &max_x);
		cv::minMaxIdx(ys_three, &min_y, &max_y);

		triangle_points[6] = max_x;
		triangle_points[7] = max_y;

		triangle_points[8] = min_x;
		triangle_points[9] = min_y;

		destination_points.push_back(triangle_points);
		
	}

	double max_x;
	double max_y;

	min_x = in_min_x;
	min_y = in_min_y;

	max_x = in_max_x;
	max_y = in_max_y;

	int w = (int)(max_x - min_x + 1.5);
    int h = (int)(max_y - min_y + 1.5);
    
	// Round the min_x and min_y for simplicity?

    pixel_mask = cv::Mat_<uchar>(h, w, (uchar)0);
    triangle_id = cv::Mat_<int>(h, w, -1);
        
	int curr_tri = -1;

	for(int y = 0; y < pixel_mask.rows; y++)
	{
		for(int x = 0; x < pixel_mask.cols; x++)
		{
			curr_tri = findTriangle(cv::Point_<double>(x + min_x, y + min_y), destination_points, curr_tri);
			// If there is a triangle at this location
            if(curr_tri != -1)
			{
				triangle_id.at<int>(y, x) = curr_tri;
                pixel_mask.at<uchar>(y, x) = 1;
			}	
		}
	}    	

	// Preallocate maps and coefficients
	coefficients.create(num_tris, 6);
	map_x.create(pixel_mask.rows,pixel_mask.cols);
	map_y.create(pixel_mask.rows,pixel_mask.cols);

}

//===========================================================================
void PAW::Read(std::ifstream& stream)
{

	stream.read ((char*)&number_of_pixels, 4);
	stream.read ((char*)&min_x, 8);
	stream.read ((char*)&min_y, 8);

	LandmarkDetector::ReadMatBin(stream, destination_landmarks);

	LandmarkDetector::ReadMatBin(stream, triangulation);

	LandmarkDetector::ReadMatBin(stream, triangle_id);
	
	cv::Mat tmpMask;	
	LandmarkDetector::ReadMatBin(stream, tmpMask);	
	tmpMask.convertTo(pixel_mask, CV_8U);	
	
	LandmarkDetector::ReadMatBin(stream, alpha);

	LandmarkDetector::ReadMatBin(stream, beta);

	map_x.create(pixel_mask.rows,pixel_mask.cols);
	map_y.create(pixel_mask.rows,pixel_mask.cols);

	coefficients.create(this->NumberOfTriangles(),6);
	
	source_landmarks = destination_landmarks;
}

//=============================================================================
// cropping from the source image to the destination image using the shape in s, used to determine if shape fitting converged successfully
void PAW::Warp(const cv::Mat& image_to_warp, cv::Mat& destination_image, const cv::Mat_<double>& landmarks_to_warp)
{
  
	// set the current shape
	source_landmarks = landmarks_to_warp.clone();

	// prepare the mapping coefficients using the current shape
	this->CalcCoeff();

	// Do the actual mapping computation (where to warp from)
	this->WarpRegion(map_x, map_y);
  	
	// Do the actual warp (with bi-linear interpolation)
	remap(image_to_warp, destination_image, map_x, map_y, CV_INTER_LINEAR);
  
}


//=============================================================================
// Calculate the warping coefficients
void PAW::CalcCoeff()
{
	int p = this->NumberOfLandmarks();

	for(int l = 0; l < this->NumberOfTriangles(); l++)
	{
	  
		int i = triangulation.at<int>(l,0);
		int j = triangulation.at<int>(l,1);
		int k = triangulation.at<int>(l,2);

		double c1 = source_landmarks.at<double>(i    , 0);
		double c2 = source_landmarks.at<double>(j    , 0) - c1;
		double c3 = source_landmarks.at<double>(k    , 0) - c1;
		double c4 = source_landmarks.at<double>(i + p, 0);
		double c5 = source_landmarks.at<double>(j + p, 0) - c4;
		double c6 = source_landmarks.at<double>(k + p, 0) - c4;

		// Get a pointer to the coefficient we will be precomputing
		double *coeff = coefficients.ptr<double>(l);

		// Extract the relevant alphas and betas
		double *c_alpha = alpha.ptr<double>(l);
		double *c_beta  = beta.ptr<double>(l);

		coeff[0] = c1 + c2 * c_alpha[0] + c3 * c_beta[0];
		coeff[1] =      c2 * c_alpha[1] + c3 * c_beta[1];
		coeff[2] =      c2 * c_alpha[2] + c3 * c_beta[2];
		coeff[3] = c4 + c5 * c_alpha[0] + c6 * c_beta[0];
		coeff[4] =      c5 * c_alpha[1] + c6 * c_beta[1];
		coeff[5] =      c5 * c_alpha[2] + c6 * c_beta[2];
	}
}

//======================================================================
// Compute the mapping coefficients
void PAW::WarpRegion(cv::Mat_<float>& mapx, cv::Mat_<float>& mapy)
{
	
	cv::MatIterator_<float> xp = mapx.begin();
	cv::MatIterator_<float> yp = mapy.begin();
	cv::MatIterator_<uchar> mp = pixel_mask.begin();
	cv::MatIterator_<int>   tp = triangle_id.begin();
	
	// The coefficients corresponding to the current triangle
	double * a;

	// Current triangle being processed	
	int k=-1;

	for(int y = 0; y < pixel_mask.rows; y++)
	{
		double yi = double(y) + min_y;
	
		for(int x = 0; x < pixel_mask.cols; x++)
		{
			double xi = double(x) + min_x;

			if(*mp == 0)
			{
				*xp = -1;
				*yp = -1;
			}
			else
			{
				// triangle corresponding to the current pixel
				int j = *tp;

				// If it is different from the previous triangle point to new coefficients
				// This will always be the case in the first iteration, hence a will not point to nothing
				if(j != k)
				{
					// Update the coefficient pointer if a new triangle is being processed
					a = coefficients.ptr<double>(j);			
					k = j;
				}  	

				//ap is now the pointer to the coefficients
				double *ap = a;							

				//look at the first coefficient (and increment). first coefficient is an x offset
				double xo = *ap++;						
				//second coefficient is an x scale as a function of x
				xo += *ap++ * xi;						
				//third coefficient ap(2) is an x scale as a function of y
				*xp = float(xo + *ap++ * yi);			

				//then fourth coefficient ap(3) is a y offset
				double yo = *ap++;						
				//fifth coeff adds coeff[4]*x to y
				yo += *ap++ * xi;						
				//final coeff adds coeff[5]*y to y
				*yp = float(yo + *ap++ * yi);			

			}
			mp++; tp++; xp++; yp++;	
		}
	}
}

// ============================================================
// Helper functions to determine which point a triangle lies in
// ============================================================

// Is the point (x0,y0) on same side as a half-plane defined by (x1,y1), (x2, y2), and (x3, y3)
bool sameSide(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3)
{
    
    double x = (x3-x2)*(y0-y2) - (x0-x2)*(y3-y2);
    double y = (x3-x2)*(y1-y2) - (x1-x2)*(y3-y2);

    return x*y >= 0;

}

// if point (x0, y0) is on same side for all three half-planes it is in a triangle
bool pointInTriangle(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3)
{
	bool same_1 = sameSide(x0, y0, x1, y1, x2, y2, x3, y3);
	bool same_2 = sameSide(x0, y0, x2, y2, x1, y1, x3, y3);
	bool same_3 = sameSide(x0, y0, x3, y3, x1, y1, x2, y2);

	return same_1 && same_2 && same_3;

}

// Find if a given point lies in the triangles
int PAW::findTriangle(const cv::Point_<double>& point, const std::vector<std::vector<double>>& control_points, int guess) const
{
    
	int num_tris = control_points.size();
	
	int tri = -1;
    
	double x0 = point.x;
	double y0 = point.y;

	// Allow a guess for speed (so as not to go through all triangles)
	if(guess != -1)
	{
		
		bool in_triangle = pointInTriangle(x0, y0, control_points[guess][0], control_points[guess][1], control_points[guess][2], control_points[guess][3], control_points[guess][4], control_points[guess][5]);
		if(in_triangle)
		{
			return guess;
		}
	}


    for (int i = 0; i < num_tris; ++i)
	{

		double max_x = control_points[i][6];
		double max_y = control_points[i][7];

		double min_x = control_points[i][8];
		double min_y = control_points[i][9];

		// Skip the check if the point is outside the bounding box of the triangle

		if( max_x < x0 || min_x > x0 || max_y < y0 || min_y > y0)
		{
			continue;
		}

		bool in_triangle = pointInTriangle(x0, y0, 
			control_points[i][0], control_points[i][1],
			control_points[i][2], control_points[i][3],
			control_points[i][4], control_points[i][5]);

        if(in_triangle)
		{
           tri = i;
           break;
		}        
	}
	return tri;
}
