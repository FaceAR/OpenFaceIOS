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

#include "Patch_experts.h"

// OpenCV includes
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>

// TBB includes
//#include <tbb/tbb.h>

// Math includes
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;

// A copy constructor
Patch_experts::Patch_experts(const Patch_experts& other) : patch_scaling(other.patch_scaling), centers(other.centers), svr_expert_intensity(other.svr_expert_intensity), svr_expert_depth(other.svr_expert_depth), ccnf_expert_intensity(other.ccnf_expert_intensity)
{

	// Make sure the matrices are allocated properly
	this->sigma_components.resize(other.sigma_components.size());
	for (size_t i = 0; i < other.sigma_components.size(); ++i)
	{
		this->sigma_components[i].resize(other.sigma_components[i].size());

		for (size_t j = 0; j < other.sigma_components[i].size(); ++j)
		{
			// Make sure the matrix is copied.
			this->sigma_components[i][j] = other.sigma_components[i][j].clone();
		}
	}

	// Make sure the matrices are allocated properly
	this->visibilities.resize(other.visibilities.size());
	for (size_t i = 0; i < other.visibilities.size(); ++i)
	{
		this->visibilities[i].resize(other.visibilities[i].size());

		for (size_t j = 0; j < other.visibilities[i].size(); ++j)
		{
			// Make sure the matrix is copied.
			this->visibilities[i][j] = other.visibilities[i][j].clone();
		}
	}
}

// Returns the patch expert responses given a grayscale and an optional depth image.
// Additionally returns the transform from the image coordinates to the response coordinates (and vice versa).
// The computation also requires the current landmark locations to compute response around, the PDM corresponding to the desired model, and the parameters describing its instance
// Also need to provide the size of the area of interest and the desired scale of analysis
void Patch_experts::Response(std::vector<cv::Mat_<float> >& patch_expert_responses, cv::Matx22f& sim_ref_to_img, cv::Matx22d& sim_img_to_ref, const cv::Mat_<uchar>& grayscale_image, const cv::Mat_<float>& depth_image,
							 const PDM& pdm, const cv::Vec6d& params_global, const cv::Mat_<double>& params_local, int window_size, int scale)
{

	int view_id = GetViewIdx(params_global, scale);		

	int n = pdm.NumberOfPoints();
		
	// Compute the current landmark locations (around which responses will be computed)
	cv::Mat_<double> landmark_locations;

	pdm.CalcShape2D(landmark_locations, params_local, params_global);

	cv::Mat_<double> reference_shape;
		
	// Initialise the reference shape on which we'll be warping
	cv::Vec6d global_ref(patch_scaling[scale], 0, 0, 0, 0, 0);

	// Compute the reference shape
	pdm.CalcShape2D(reference_shape, params_local, global_ref);
		
	// similarity and inverse similarity transform to and from image and reference shape
	cv::Mat_<double> reference_shape_2D = (reference_shape.reshape(1, 2).t());
	cv::Mat_<double> image_shape_2D = landmark_locations.reshape(1, 2).t();

	sim_img_to_ref = AlignShapesWithScale(image_shape_2D, reference_shape_2D);
	cv::Matx22d sim_ref_to_img_d = sim_img_to_ref.inv(cv::DECOMP_LU);

	double a1 = sim_ref_to_img_d(0,0);
	double b1 = -sim_ref_to_img_d(0,1);
		
	sim_ref_to_img(0,0) = (float)sim_ref_to_img_d(0,0);
	sim_ref_to_img(0,1) = (float)sim_ref_to_img_d(0,1);
	sim_ref_to_img(1,0) = (float)sim_ref_to_img_d(1,0);
	sim_ref_to_img(1,1) = (float)sim_ref_to_img_d(1,1);

	// Indicates the legal pixels in a depth image, if available (used for CLM-Z area of interest (window) interpolation)
	cv::Mat_<uchar> mask;
	if(!depth_image.empty())
	{
		mask = depth_image > 0;			
		mask = mask / 255;
	}		
	

	bool use_ccnf = !this->ccnf_expert_intensity.empty();

	// If using CCNF patch experts might need to precalculate Sigmas
	if(use_ccnf)
	{
        std::vector<cv::Mat_<float> > sigma_components;

		// Retrieve the correct sigma component size
		for( size_t w_size = 0; w_size < this->sigma_components.size(); ++w_size)
		{
			if(!this->sigma_components[w_size].empty())
			{
				if(window_size*window_size == this->sigma_components[w_size][0].rows)
				{
					sigma_components = this->sigma_components[w_size];
				}
			}
		}			

		// Go through all of the landmarks and compute the Sigma for each
		for( int lmark = 0; lmark < n; lmark++)
		{
			// Only for visible landmarks
			if(visibilities[scale][view_id].at<int>(lmark,0))
			{
				// Precompute sigmas if they are not computed yet
				ccnf_expert_intensity[scale][view_id][lmark].ComputeSigmas(sigma_components, window_size);
			}
		}

	}

	// calculate the patch responses for every landmark, Actual work happens here. If openMP is turned on it is possible to do this in parallel,
	// this might work well on some machines, while potentially have an adverse effect on others
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
//	tbb::parallel_for(0, (int)n, [&](int i){
    for(int i = 0; i < n; i++)
	{
			
		if(visibilities[scale][view_id].rows == n)
		{
			if(visibilities[scale][view_id].at<int>(i,0) != 0)
			{

				// Work out how big the area of interest has to be to get a response of window size
				int area_of_interest_width;
				int area_of_interest_height;

				if(use_ccnf)
				{
					area_of_interest_width = window_size + ccnf_expert_intensity[scale][view_id][i].width - 1; 
					area_of_interest_height = window_size + ccnf_expert_intensity[scale][view_id][i].height - 1;				
				}
				else
				{
					area_of_interest_width = window_size + svr_expert_intensity[scale][view_id][i].width - 1; 
					area_of_interest_height = window_size + svr_expert_intensity[scale][view_id][i].height - 1;
				}
			
				// scale and rotate to mean shape to reference frame
				cv::Mat sim = (cv::Mat_<float>(2,3) << a1, -b1, landmark_locations.at<double>(i,0), b1, a1, landmark_locations.at<double>(i+n,0));

				// Extract the region of interest around the current landmark location
				cv::Mat_<float> area_of_interest(area_of_interest_height, area_of_interest_width);

				// Using C style openCV as it does what we need
				CvMat area_of_interest_o = area_of_interest;
				CvMat sim_o = sim;
				IplImage im_o = grayscale_image;			
				cvGetQuadrangleSubPix(&im_o, &area_of_interest_o, &sim_o);
			
				// get the correct size response window			
				patch_expert_responses[i] = cv::Mat_<float>(window_size, window_size);

				// Get intensity response either from the SVR or CCNF patch experts (prefer CCNF)
				if(!ccnf_expert_intensity.empty())
				{				

					ccnf_expert_intensity[scale][view_id][i].Response(area_of_interest, patch_expert_responses[i]);
				}
				else
				{
					svr_expert_intensity[scale][view_id][i].Response(area_of_interest, patch_expert_responses[i]);
				}
			
				// if we have a corresponding depth patch and it is visible		
				if(!svr_expert_depth.empty() && !depth_image.empty() && visibilities[scale][view_id].at<int>(i,0))
				{

					cv::Mat_<float> dProb = patch_expert_responses[i].clone();
					cv::Mat_<float> depthWindow(area_of_interest_height, area_of_interest_width);
			

					CvMat dimg_o = depthWindow;
					cv::Mat maskWindow(area_of_interest_height, area_of_interest_width, CV_32F);
					CvMat mimg_o = maskWindow;

					IplImage d_o = depth_image;
					IplImage m_o = mask;

					cvGetQuadrangleSubPix(&d_o,&dimg_o,&sim_o);
				
					cvGetQuadrangleSubPix(&m_o,&mimg_o,&sim_o);

					depthWindow.setTo(0, maskWindow < 1);

					svr_expert_depth[scale][view_id][i].ResponseDepth(depthWindow, dProb);
							
					// Sum to one
					double sum = cv::sum(patch_expert_responses[i])[0];

					// To avoid division by 0 issues
					if(sum == 0)
					{
						sum = 1;
					}

					patch_expert_responses[i] /= sum;

					// Sum to one
					sum = cv::sum(dProb)[0];
					// To avoid division by 0 issues
					if(sum == 0)
					{
						sum = 1;
					}

					dProb /= sum;

					patch_expert_responses[i] = patch_expert_responses[i] + dProb;

				}
			}
		}
	}
//	});

}

//=============================================================================
// Getting the closest view center based on orientation
int Patch_experts::GetViewIdx(const cv::Vec6d& params_global, int scale) const
{	
	int idx = 0;
	
	double dbest;

	for(int i = 0; i < this->nViews(scale); i++)
	{
		double v1 = params_global[1] - centers[scale][i][0]; 
		double v2 = params_global[2] - centers[scale][i][1];
		double v3 = params_global[3] - centers[scale][i][2];
			
		double d = v1*v1 + v2*v2 + v3*v3;

		if(i == 0 || d < dbest)
		{
			dbest = d;
			idx = i;
		}
	}
	return idx;
}


//===========================================================================
void Patch_experts::Read(std::vector<std::string> intensity_svr_expert_locations, std::vector<std::string> depth_svr_expert_locations, std::vector<std::string> intensity_ccnf_expert_locations)
{

	// initialise the SVR intensity patch expert parameters
	int num_intensity_svr = intensity_svr_expert_locations.size();
	centers.resize(num_intensity_svr);
	visibilities.resize(num_intensity_svr);
	patch_scaling.resize(num_intensity_svr);
	
	svr_expert_intensity.resize(num_intensity_svr);
	
	// Reading in SVR intensity patch experts for each scales it is defined in
	for(int scale = 0; scale < num_intensity_svr; ++scale)
	{		
        std::string location = intensity_svr_expert_locations[scale];
        std::cout << "Reading the intensity SVR patch experts from: " << location << "....";
		Read_SVR_patch_experts(location,  centers[scale], visibilities[scale], svr_expert_intensity[scale], patch_scaling[scale]);
	}

	// Initialise and read CCNF patch experts (currently only intensity based), 
	int num_intensity_ccnf = intensity_ccnf_expert_locations.size();

	// CCNF experts override the SVR ones
	if(num_intensity_ccnf > 0)
	{
		centers.resize(num_intensity_ccnf);
		visibilities.resize(num_intensity_ccnf);
		patch_scaling.resize(num_intensity_ccnf);
		ccnf_expert_intensity.resize(num_intensity_ccnf);
	}

	for(int scale = 0; scale < num_intensity_ccnf; ++scale)
	{		
        std::string location = intensity_ccnf_expert_locations[scale];
        std::cout << "Reading the intensity CCNF patch experts from: " << location << "....";
		Read_CCNF_patch_experts(location,  centers[scale], visibilities[scale], ccnf_expert_intensity[scale], patch_scaling[scale]);
	}


	// initialise the SVR depth patch expert parameters
	int num_depth_scales = depth_svr_expert_locations.size();
	int num_intensity_scales = centers.size();
	
	if(num_depth_scales > 0 && num_intensity_scales != num_depth_scales)
	{
        std::cout << "Intensity and depth patch experts have a different number of scales, can't read depth" << std::endl;
		return;
	}

	// Have these to confirm that depth patch experts have the same number of views and scales and have the same visibilities
    std::vector<std::vector<cv::Vec3d> > centers_depth(num_depth_scales);
    std::vector<std::vector<cv::Mat_<int> > > visibilities_depth(num_depth_scales);
    std::vector<double> patch_scaling_depth(num_depth_scales);
	
	svr_expert_depth.resize(num_depth_scales);	

	// Reading in SVR intensity patch experts for each scales it is defined in
	for(int scale = 0; scale < num_depth_scales; ++scale)
	{		
        std::string location = depth_svr_expert_locations[scale];
        std::cout << "Reading the depth SVR patch experts from: " << location << "....";
		Read_SVR_patch_experts(location,  centers_depth[scale], visibilities_depth[scale], svr_expert_depth[scale], patch_scaling_depth[scale]);

		// Check if the scales are identical
		if(patch_scaling_depth[scale] != patch_scaling[scale])
		{
            std::cout << "Intensity and depth patch experts have a different scales, can't read depth" << std::endl;
			svr_expert_depth.clear();
			return;			
		}

		int num_views_intensity = centers[scale].size();
		int num_views_depth = centers_depth[scale].size();

		// Check if the number of views is identical
		if(num_views_intensity != num_views_depth)
		{
            std::cout << "Intensity and depth patch experts have a different number of scales, can't read depth" << std::endl;
			svr_expert_depth.clear();
			return;			
		}

		for(int view = 0; view < num_views_depth; ++view)
		{
			if(cv::countNonZero(centers_depth[scale][view] != centers[scale][view]) || cv::countNonZero(visibilities[scale][view] != visibilities_depth[scale][view]))
			{
                std::cout << "Intensity and depth patch experts have different visibilities or centers" << std::endl;
				svr_expert_depth.clear();
				return;		
			}
		}
	}

}
//======================= Reading the SVR patch experts =========================================//
void Patch_experts::Read_SVR_patch_experts(std::string expert_location, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<Multi_SVR_patch_expert> >& patches, double& scale)
{

    std::ifstream patchesFile(expert_location.c_str(), std::ios_base::in);

	if(patchesFile.is_open())
	{
		LandmarkDetector::SkipComments(patchesFile);

		patchesFile >> scale;

		LandmarkDetector::SkipComments(patchesFile);

		int numberViews;		

		patchesFile >> numberViews; 

		// read the visibility
		centers.resize(numberViews);
		visibility.resize(numberViews);
  
		patches.resize(numberViews);

		LandmarkDetector::SkipComments(patchesFile);

		// centers of each view (which view corresponds to which orientation)
		for(size_t i = 0; i < centers.size(); i++)
		{
			cv::Mat center;
			LandmarkDetector::ReadMat(patchesFile, center);	
			center.copyTo(centers[i]);
			centers[i] = centers[i] * M_PI / 180.0;
		}

		LandmarkDetector::SkipComments(patchesFile);

		// the visibility of points for each of the views (which verts are visible at a specific view
		for(size_t i = 0; i < visibility.size(); i++)
		{
			LandmarkDetector::ReadMat(patchesFile, visibility[i]);				
		}

		int numberOfPoints = visibility[0].rows;

		LandmarkDetector::SkipComments(patchesFile);

		// read the patches themselves
		for(size_t i = 0; i < patches.size(); i++)
		{
			// number of patches for each view
			patches[i].resize(numberOfPoints);
			// read in each patch
			for(int j = 0; j < numberOfPoints; j++)
			{
				patches[i][j].Read(patchesFile);
			}
		}
	
        std::cout << "Done" << std::endl;
	}
	else
	{
        std::cout << "Can't find/open the patches file" << std::endl;
	}
}

//======================= Reading the CCNF patch experts =========================================//
void Patch_experts::Read_CCNF_patch_experts(std::string patchesFileLocation, std::vector<cv::Vec3d>& centers, std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<CCNF_patch_expert> >& patches, double& patchScaling)
{

    std::ifstream patchesFile(patchesFileLocation.c_str(), std::ios::in | std::ios::binary);

	if(patchesFile.is_open())
	{
		patchesFile.read ((char*)&patchScaling, 8);
		
		int numberViews;		
		patchesFile.read ((char*)&numberViews, 4);

		// read the visibility
		centers.resize(numberViews);
		visibility.resize(numberViews);
  
		patches.resize(numberViews);
		
		// centers of each view (which view corresponds to which orientation)
		for(size_t i = 0; i < centers.size(); i++)
		{
			cv::Mat center;
			LandmarkDetector::ReadMatBin(patchesFile, center);	
			center.copyTo(centers[i]);
			centers[i] = centers[i] * M_PI / 180.0;
		}

		// the visibility of points for each of the views (which verts are visible at a specific view
		for(size_t i = 0; i < visibility.size(); i++)
		{
			LandmarkDetector::ReadMatBin(patchesFile, visibility[i]);				
		}
		int numberOfPoints = visibility[0].rows;

		// Read the possible SigmaInvs (without beta), this will be followed by patch reading (this assumes all of them have the same type, and number of betas)
		int num_win_sizes;
		int num_sigma_comp;
		patchesFile.read ((char*)&num_win_sizes, 4);

        std::vector<int> windows;
		windows.resize(num_win_sizes);

        std::vector<std::vector<cv::Mat_<float> > > sigma_components;
		sigma_components.resize(num_win_sizes);

		for (int w=0; w < num_win_sizes; ++w)
		{
			patchesFile.read ((char*)&windows[w], 4);

			patchesFile.read ((char*)&num_sigma_comp, 4);

			sigma_components[w].resize(num_sigma_comp);

			for(int s=0; s < num_sigma_comp; ++s)
			{
				LandmarkDetector::ReadMatBin(patchesFile, sigma_components[w][s]);
			}
		}
		
		this->sigma_components = sigma_components;

		// read the patches themselves
		for(size_t i = 0; i < patches.size(); i++)
		{
			// number of patches for each view
			patches[i].resize(numberOfPoints);
			// read in each patch
			for(int j = 0; j < numberOfPoints; j++)
			{
				patches[i][j].Read(patchesFile, windows, sigma_components);
			}
		}
        std::cout << "Done" << std::endl;
	}
	else
	{
        std::cout << "Can't find/open the patches file" << std::endl;
	}
}


