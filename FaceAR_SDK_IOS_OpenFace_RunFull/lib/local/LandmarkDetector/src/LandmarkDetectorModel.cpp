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

#include "LandmarkDetectorModel.h"

//// Boost includes
//#include <filesystem.hpp>
//#include <filesystem/fstream.hpp>

// TBB includes
//#include <tbb/tbb.h>

// Local includes
#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;

//=============================================================================
//=============================================================================

// Constructors
// A default constructor
CLNF::CLNF()
{
//    FaceModelParameters parameters;
//    this->Read(parameters.model_location);
}

bool CLNF::inits()
{
    FaceModelParameters parameters;
    parameters.init();
    parameters.model_location = model_location_clnf;
    parameters.face_detector_location = face_detector_location_clnf;
    
    this->Read(parameters.model_location);
    return true;
}

// Constructor from a model file
CLNF::CLNF(std::string fname)
{
    this->Read(fname);
}

// Copy constructor (makes a deep copy of CLNF)
CLNF::CLNF(const CLNF& other): pdm(other.pdm), params_local(other.params_local.clone()), params_global(other.params_global), detected_landmarks(other.detected_landmarks.clone()),
    landmark_likelihoods(other.landmark_likelihoods.clone()), patch_experts(other.patch_experts), landmark_validator(other.landmark_validator), face_detector_location(other.face_detector_location),
    hierarchical_mapping(other.hierarchical_mapping), hierarchical_models(other.hierarchical_models), hierarchical_model_names(other.hierarchical_model_names),
    hierarchical_params(other.hierarchical_params), eye_model(other.eye_model)
{
    this->detection_success = other.detection_success;
    this->tracking_initialised = other.tracking_initialised;
    this->detection_certainty = other.detection_certainty;
    this->model_likelihood = other.model_likelihood;
    this->failures_in_a_row = other.failures_in_a_row;

    // Load the CascadeClassifier (as it does not have a proper copy constructor)
    if(!face_detector_location.empty())
    {
        this->face_detector_HAAR.load(face_detector_location);
    }
    // Make sure the matrices are allocated properly
    this->triangulations.resize(other.triangulations.size());
    for(size_t i = 0; i < other.triangulations.size(); ++i)
    {
        // Make sure the matrix is copied.
        this->triangulations[i] = other.triangulations[i].clone();
    }

    // Make sure the matrices are allocated properly
    for(std::map<int, cv::Mat_<float>>::const_iterator it = other.kde_resp_precalc.begin(); it!= other.kde_resp_precalc.end(); it++)
    {
        // Make sure the matrix is copied.
        this->kde_resp_precalc.insert(std::pair<int, cv::Mat_<float>>(it->first, it->second.clone()));
    }

//    this->face_detector_HOG = dlib::get_frontal_face_detector();

}

// Assignment operator for lvalues (makes a deep copy of CLNF)
CLNF & CLNF::operator= (const CLNF& other)
{
    if (this != &other) // protect against invalid self-assignment
    {
        pdm = PDM(other.pdm);
        params_local = other.params_local.clone();
        params_global = other.params_global;
        detected_landmarks = other.detected_landmarks.clone();

        landmark_likelihoods =other.landmark_likelihoods.clone();
        patch_experts = Patch_experts(other.patch_experts);
        landmark_validator = DetectionValidator(other.landmark_validator);
        face_detector_location = other.face_detector_location;

        this->detection_success = other.detection_success;
        this->tracking_initialised = other.tracking_initialised;
        this->detection_certainty = other.detection_certainty;
        this->model_likelihood = other.model_likelihood;
        this->failures_in_a_row = other.failures_in_a_row;

        this->eye_model = other.eye_model;

        // Load the CascadeClassifier (as it does not have a proper copy constructor)
        if(!face_detector_location.empty())
        {
            this->face_detector_HAAR.load(face_detector_location);
        }
        // Make sure the matrices are allocated properly
        this->triangulations.resize(other.triangulations.size());
        for(size_t i = 0; i < other.triangulations.size(); ++i)
        {
            // Make sure the matrix is copied.
            this->triangulations[i] = other.triangulations[i].clone();
        }

        // Make sure the matrices are allocated properly
        for(std::map<int, cv::Mat_<float>>::const_iterator it = other.kde_resp_precalc.begin(); it!= other.kde_resp_precalc.end(); it++)
        {
            // Make sure the matrix is copied.
            this->kde_resp_precalc.insert(std::pair<int, cv::Mat_<float>>(it->first, it->second.clone()));
        }

        // Copy over the hierarchical models
        this->hierarchical_mapping = other.hierarchical_mapping;
        this->hierarchical_models = other.hierarchical_models;
        this->hierarchical_model_names = other.hierarchical_model_names;
        this->hierarchical_params = other.hierarchical_params;
    }

//    face_detector_HOG = dlib::get_frontal_face_detector();

    return *this;
}

// Move constructor
CLNF::CLNF(const CLNF&& other)
{
    this->detection_success = other.detection_success;
    this->tracking_initialised = other.tracking_initialised;
    this->detection_certainty = other.detection_certainty;
    this->model_likelihood = other.model_likelihood;
    this->failures_in_a_row = other.failures_in_a_row;

    pdm = other.pdm;
    params_local = other.params_local;
    params_global = other.params_global;
    detected_landmarks = other.detected_landmarks;
    landmark_likelihoods = other.landmark_likelihoods;
    patch_experts = other.patch_experts;
    landmark_validator = other.landmark_validator;
    face_detector_location = other.face_detector_location;

    face_detector_HAAR = other.face_detector_HAAR;

    triangulations = other.triangulations;
    kde_resp_precalc = other.kde_resp_precalc;

//    face_detector_HOG = dlib::get_frontal_face_detector();

    // Copy over the hierarchical models
    this->hierarchical_mapping = other.hierarchical_mapping;
    this->hierarchical_models = other.hierarchical_models;
    this->hierarchical_model_names = other.hierarchical_model_names;
    this->hierarchical_params = other.hierarchical_params;

    this->eye_model = other.eye_model;

}

// Assignment operator for rvalues
CLNF & CLNF::operator= (const CLNF&& other)
{
    this->detection_success = other.detection_success;
    this->tracking_initialised = other.tracking_initialised;
    this->detection_certainty = other.detection_certainty;
    this->model_likelihood = other.model_likelihood;
    this->failures_in_a_row = other.failures_in_a_row;

    pdm = other.pdm;
    params_local = other.params_local;
    params_global = other.params_global;
    detected_landmarks = other.detected_landmarks;
    landmark_likelihoods = other.landmark_likelihoods;
    patch_experts = other.patch_experts;
    landmark_validator = other.landmark_validator;
    face_detector_location = other.face_detector_location;

    face_detector_HAAR = other.face_detector_HAAR;

    triangulations = other.triangulations;
    kde_resp_precalc = other.kde_resp_precalc;

//    face_detector_HOG = dlib::get_frontal_face_detector();

    // Copy over the hierarchical models
    this->hierarchical_mapping = other.hierarchical_mapping;
    this->hierarchical_models = other.hierarchical_models;
    this->hierarchical_model_names = other.hierarchical_model_names;
    this->hierarchical_params = other.hierarchical_params;

    this->eye_model = other.eye_model;

    return *this;
}


void CLNF::Read_CLNF(std::string clnf_location)
{
    std::cout << "clnf_location = " << clnf_location << std::endl;
    // Location of modules
    std::ifstream locations(clnf_location.c_str(), std::ios_base::in);

    if(!locations.is_open())
    {
        std::cout << "Couldn't open the CLNF model file aborting" << std::endl;
        std::cout.flush();
        return;
    }

    std::string line;

    std::vector<std::string> intensity_expert_locations;
    std::vector<std::string> depth_expert_locations;
    std::vector<std::string> ccnf_expert_locations;

    // The other module locations should be defined as relative paths from the main model
//    boost::filesystem::path root = boost::filesystem::path(clnf_location).parent_path();
    std::string root;
    std::string temp("/");
    std::size_t found = clnf_location.rfind(temp);
    if (found!=std::string::npos)
        root = clnf_location.substr(0, found);

    // The main file contains the references to other files
    while (!locations.eof())
    {

        getline(locations, line);

        std::stringstream lineStream(line);

        std::string module;
        std::string location;

        // figure out which module is to be read from which file
        lineStream >> module;

        getline(lineStream, location);

        if(location.size() > 0)
            location.erase(location.begin()); // remove the first space

        // remove carriage return at the end for compatibility with unix systems
        if(location.size() > 0 && location.at(location.size()-1) == '\r')
        {
            location = location.substr(0, location.size()-1);
        }

        // append the lovstion to root location (boost syntax)
//        location = (root / location).string();
        location = root + temp + location;

        if (module.compare("PDM") == 0)
        {
            std::cout << "Reading the PDM module from: " << location << "....";
            pdm.Read(location);

            std::cout << "Done" << std::endl;
        }
        else if (module.compare("Triangulations") == 0)
        {
            std::cout << "Reading the Triangulations module from: " << location << "....";
            std::ifstream triangulationFile(location.c_str(), std::ios_base::in);

            LandmarkDetector::SkipComments(triangulationFile);

            int numViews;
            triangulationFile >> numViews;

            // read in the triangulations
            triangulations.resize(numViews);

            for(int i = 0; i < numViews; ++i)
            {
                LandmarkDetector::SkipComments(triangulationFile);
                LandmarkDetector::ReadMat(triangulationFile, triangulations[i]);
            }
            std::cout << "Done" << std::endl;
        }
        else if(module.compare("PatchesIntensity") == 0)
        {
            intensity_expert_locations.push_back(location);
        }
        else if(module.compare("PatchesDepth") == 0)
        {
            depth_expert_locations.push_back(location);
        }
        else if(module.compare("PatchesCCNF") == 0)
        {
            ccnf_expert_locations.push_back(location);
        }
    }

    // Initialise the patch experts
    patch_experts.Read(intensity_expert_locations, depth_expert_locations, ccnf_expert_locations);

    // Read in a face detector
//    face_detector_HOG = dlib::get_frontal_face_detector();

}

void CLNF::Read(std::string main_location)
{
    std::cout << "Reading the CLNF landmark detector/tracker from: " << main_location << std::endl;

    std::ifstream locations(main_location.c_str(), std::ios_base::in);
    if(!locations.is_open())
    {
        std::cout << "Couldn't open the model file, aborting" << std::endl;
        return;
    }
    std::string line;

    // The other module locations should be defined as relative paths from the main model
//    boost::filesystem::path root = boost::filesystem::path(main_location).parent_path();
    std::string root;
    std::string temp("/");
    std::size_t found = main_location.rfind(temp);
    if (found!=std::string::npos)
        root = main_location.substr(0, found);
      //std::cout << "first '/' found at: " << found << '\n';

    // The main file contains the references to other files
    while (!locations.eof())
    {
        getline(locations, line);

        std::stringstream lineStream(line);

        std::string module;
        std::string location;

        // figure out which module is to be read from which file
        lineStream >> module;

        lineStream >> location;

        // remove carriage return at the end for compatibility with unix systems
        if(location.size() > 0 && location.at(location.size()-1) == '\r')
        {
            location = location.substr(0, location.size()-1);
        }

        // append to root
//        location = (root / location).string();
        location = root + temp + location;

        if (module.compare("LandmarkDetector") == 0)
        {
            std::cout << "Reading the landmark detector module from: " << location << std::endl;

            // The CLNF module includes the PDM and the patch experts
            Read_CLNF(location);
        }
        else if(module.compare("LandmarkDetector_part") == 0)
        {
            std::string part_name;
            lineStream >> part_name;
            std::cout << "Reading part based module...." << part_name << std::endl;

            std::vector<std::pair<int, int>> mappings;
            while(!lineStream.eof())
            {
                int ind_in_main;
                lineStream >> ind_in_main;

                int ind_in_part;
                lineStream >> ind_in_part;
                mappings.push_back(std::pair<int, int>(ind_in_main, ind_in_part));
            }

            this->hierarchical_mapping.push_back(mappings);

            CLNF part_model(location);

            this->hierarchical_models.push_back(part_model);

            this->hierarchical_model_names.push_back(part_name);

            FaceModelParameters params;
            params.validate_detections = false;
            params.refine_hierarchical = false;
            params.refine_parameters = false;

            if(part_name.compare("left_eye") == 0 || part_name.compare("right_eye") == 0)
            {

                std::vector<int> windows_large;
                windows_large.push_back(5);
                windows_large.push_back(3);

                std::vector<int> windows_small;
                windows_small.push_back(5);
                windows_small.push_back(3);

                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;

                params.reg_factor = 0.1;
                params.sigma = 2;
            }
            else if(part_name.compare("left_eye_28") == 0 || part_name.compare("right_eye_28") == 0)
            {
                std::vector<int> windows_large;
                windows_large.push_back(3);
                windows_large.push_back(5);
                windows_large.push_back(9);

                std::vector<int> windows_small;
                windows_small.push_back(3);
                windows_small.push_back(5);
                windows_small.push_back(9);

                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;

                params.reg_factor = 0.5;
                params.sigma = 1.0;

                eye_model = true;

            }
            else if(part_name.compare("mouth") == 0)
            {
                std::vector<int> windows_large;
                windows_large.push_back(7);
                windows_large.push_back(7);

                std::vector<int> windows_small;
                windows_small.push_back(7);
                windows_small.push_back(7);

                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;

                params.reg_factor = 1.0;
                params.sigma = 2.0;
            }
            else if(part_name.compare("brow") == 0)
            {
                std::vector<int> windows_large;
                windows_large.push_back(11);
                windows_large.push_back(9);

                std::vector<int> windows_small;
                windows_small.push_back(11);
                windows_small.push_back(9);

                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;

                params.reg_factor = 10.0;
                params.sigma = 3.5;
            }
            else if(part_name.compare("inner") == 0)
            {
                std::vector<int> windows_large;
                windows_large.push_back(9);

                std::vector<int> windows_small;
                windows_small.push_back(9);

                params.window_sizes_init = windows_large;
                params.window_sizes_small = windows_small;
                params.window_sizes_current = windows_large;

                params.reg_factor = 2.5;
                params.sigma = 1.75;
                params.weight_factor = 2.5;
            }

            this->hierarchical_params.push_back(params);

            std::cout << "Done" << std::endl;
        }
        else if (module.compare("DetectionValidator") == 0)
        {
            std::cout << "Reading the landmark validation module....";
            landmark_validator.Read(location);
            std::cout << "Done" << std::endl;
        }
    }

    detected_landmarks.create(2 * pdm.NumberOfPoints(), 1);
    detected_landmarks.setTo(0);

    detection_success = false;
    tracking_initialised = false;
    model_likelihood = -10; // very low
    detection_certainty = 1; // very uncertain

    // Initialising default values for the rest of the variables

    // local parameters (shape)
    params_local.create(pdm.NumberOfModes(), 1);
    params_local.setTo(0.0);

    // global parameters (pose) [scale, euler_x, euler_y, euler_z, tx, ty]
    params_global = cv::Vec6d(1, 0, 0, 0, 0, 0);

    failures_in_a_row = -1;

}

// Resetting the model (for a new video, or complet reinitialisation
void CLNF::Reset()
{
    detected_landmarks.setTo(0);

    detection_success = false;
    tracking_initialised = false;
    model_likelihood = -10;  // very low
    detection_certainty = 1; // very uncertain

    // local parameters (shape)
    params_local.setTo(0.0);

    // global parameters (pose) [scale, euler_x, euler_y, euler_z, tx, ty]
    params_global = cv::Vec6d(1, 0, 0, 0, 0, 0);

    failures_in_a_row = -1;
    face_template = cv::Mat_<uchar>();
}

// Resetting the model, choosing the face nearest (x,y)
void CLNF::Reset(double x, double y)
{

    // First reset the model overall
    this->Reset();

    // Now in the following frame when face detection takes place this is the point at which it will be preffered
    this->preference_det.x = x;
    this->preference_det.y = y;

}

// The main internal landmark detection call (should not be used externally?)
bool CLNF::DetectLandmarks(const cv::Mat_<uchar> &image, const cv::Mat_<float> &depth, FaceModelParameters& params)
{

    // Fits from the current estimate of local and global parameters in the model
    bool fit_success = Fit(image, depth, params.window_sizes_current, params);

    // Store the landmarks converged on in detected_landmarks
    pdm.CalcShape2D(detected_landmarks, params_local, params_global);

    if(params.refine_hierarchical && hierarchical_models.size() > 0)
    {
        bool parts_used = false;

        // Do the hierarchical models in parallel
//        tbb::parallel_for(0, (int)hierarchical_models.size(), [&](int part_model){
            for(int part_model = 0; part_model < hierarchical_models.size(); part_model++)
            {
                // Only do the synthetic eye models if we're doing gaze
                if (!((hierarchical_model_names[part_model].compare("right_eye_28") == 0 ||
                       hierarchical_model_names[part_model].compare("left_eye_28") == 0)
                      && !params.track_gaze))
                {

                    int n_part_points = hierarchical_models[part_model].pdm.NumberOfPoints();

                    std::vector<std::pair<int, int> > mappings = this->hierarchical_mapping[part_model];

                    cv::Mat_<double> part_model_locs(n_part_points * 2, 1, 0.0);

                    // Extract the corresponding landmarks
                    for (size_t mapping_ind = 0; mapping_ind < mappings.size(); ++mapping_ind)
                    {
                        part_model_locs.at<double>(mappings[mapping_ind].second) = detected_landmarks.at<double>(mappings[mapping_ind].first);
                        part_model_locs.at<double>(mappings[mapping_ind].second + n_part_points) = detected_landmarks.at<double>(mappings[mapping_ind].first + this->pdm.NumberOfPoints());
                    }

                    // Fit the part based model PDM
                    hierarchical_models[part_model].pdm.CalcParams(hierarchical_models[part_model].params_global, hierarchical_models[part_model].params_local, part_model_locs);

                    // Only do this if we don't need to upsample
                    if (params_global[0] > 0.9 * hierarchical_models[part_model].patch_experts.patch_scaling[0])
                    {
                        parts_used = true;

                        this->hierarchical_params[part_model].window_sizes_current = this->hierarchical_params[part_model].window_sizes_init;

                        // Do the actual landmark detection
                        hierarchical_models[part_model].DetectLandmarks(image, depth, hierarchical_params[part_model]);

                        // Reincorporate the models into main tracker
                        for (size_t mapping_ind = 0; mapping_ind < mappings.size(); ++mapping_ind)
                        {
                            detected_landmarks.at<double>(mappings[mapping_ind].first) = hierarchical_models[part_model].detected_landmarks.at<double>(mappings[mapping_ind].second);
                            detected_landmarks.at<double>(mappings[mapping_ind].first + pdm.NumberOfPoints()) = hierarchical_models[part_model].detected_landmarks.at<double>(mappings[mapping_ind].second + hierarchical_models[part_model].pdm.NumberOfPoints());
                        }
                    }
                    else
                    {
                        hierarchical_models[part_model].pdm.CalcShape2D(hierarchical_models[part_model].detected_landmarks, hierarchical_models[part_model].params_local, hierarchical_models[part_model].params_global);
                    }
                }
            }
//        });

        // Recompute main model based on the fit part models
        if(parts_used)
        {
            pdm.CalcParams(params_global, params_local, detected_landmarks);
            pdm.CalcShape2D(detected_landmarks, params_local, params_global);
        }
    }

    // Check detection correctness
    if(params.validate_detections && fit_success)
    {
        cv::Vec3d orientation(params_global[1], params_global[2], params_global[3]);

        detection_certainty = landmark_validator.Check(orientation, image, detected_landmarks);

        detection_success = detection_certainty < params.validation_boundary;
    }
    else
    {
        detection_success = fit_success;
        if(fit_success)
        {
            detection_certainty = -1;
        }
        else
        {
            detection_certainty = 1;
        }

    }

    return detection_success;
}

//=============================================================================
bool CLNF::Fit(const cv::Mat_<uchar>& im, const cv::Mat_<float>& depthImg, const std::vector<int>& window_sizes, const FaceModelParameters& parameters)
{
    // Making sure it is a single channel image
    assert(im.channels() == 1);

    // Placeholder for the landmarks
    cv::Mat_<double> current_shape(2 * pdm.NumberOfPoints() , 1, 0.0);

    int n = pdm.NumberOfPoints();

    cv::Mat_<float> depth_img_no_background;

    // Background elimination from the depth image
    if(!depthImg.empty())
    {
        bool success = RemoveBackground(depth_img_no_background, depthImg);

        // The attempted background removal can fail leading to tracking failure
        if(!success)
        {
            return false;
        }
    }

    int num_scales = patch_experts.patch_scaling.size();

    // Storing the patch expert response maps
    std::vector<cv::Mat_<float> > patch_expert_responses(n);

    // Converting from image space to patch expert space (normalised for rotation and scale)
    cv::Matx22f sim_ref_to_img;
    cv::Matx22d sim_img_to_ref;

    FaceModelParameters tmp_parameters = parameters;

    // Optimise the model across a number of areas of interest (usually in descending window size and ascending scale size)
    for(int scale = 0; scale < num_scales; scale++)
    {

        int window_size = window_sizes[scale];

        if(window_size == 0 ||  0.9 * patch_experts.patch_scaling[scale] > params_global[0])
            continue;

        // The patch expert response computation
        if(scale != window_sizes.size() - 1)
        {
            patch_experts.Response(patch_expert_responses, sim_ref_to_img, sim_img_to_ref, im, depth_img_no_background, pdm, params_global, params_local, window_size, scale);
        }
        else
        {
            // Do not use depth for the final iteration as it is not as accurate
            patch_experts.Response(patch_expert_responses, sim_ref_to_img, sim_img_to_ref, im, cv::Mat(), pdm, params_global, params_local, window_size, scale);
        }

        if(parameters.refine_parameters == true)
        {
            // Adapt the parameters based on scale (wan't to reduce regularisation as scale increases, but increa sigma and tikhonov)
            tmp_parameters.reg_factor = parameters.reg_factor - 15 * log(patch_experts.patch_scaling[scale]/0.25)/log(2);

            if(tmp_parameters.reg_factor <= 0)
                tmp_parameters.reg_factor = 0.001;

            tmp_parameters.sigma = parameters.sigma + 0.25 * log(patch_experts.patch_scaling[scale]/0.25)/log(2);
            tmp_parameters.weight_factor = parameters.weight_factor + 2 * parameters.weight_factor *  log(patch_experts.patch_scaling[scale]/0.25)/log(2);
        }

        // Get the current landmark locations
        pdm.CalcShape2D(current_shape, params_local, params_global);

        // Get the view used by patch experts
        int view_id = patch_experts.GetViewIdx(params_global, scale);

        // the actual optimisation step
        this->NU_RLMS(params_global, params_local, patch_expert_responses, cv::Vec6d(params_global), params_local.clone(), current_shape, sim_img_to_ref, sim_ref_to_img, window_size, view_id, true, scale, this->landmark_likelihoods, tmp_parameters);

        // non-rigid optimisation
        this->model_likelihood = this->NU_RLMS(params_global, params_local, patch_expert_responses, cv::Vec6d(params_global), params_local.clone(), current_shape, sim_img_to_ref, sim_ref_to_img, window_size, view_id, false, scale, this->landmark_likelihoods, tmp_parameters);

        // Can't track very small images reliably (less than ~30px across)
        if(params_global[0] < 0.25)
        {
            std::cout << "Face too small for landmark detection" << std::endl;
            return false;
        }
    }

    return true;
}

void CLNF::NonVectorisedMeanShift_precalc_kde(cv::Mat_<float>& out_mean_shifts, const std::vector<cv::Mat_<float> >& patch_expert_responses, const cv::Mat_<float> &dxs, const cv::Mat_<float> &dys, int resp_size, float a, int scale, int view_id, std::map<int, cv::Mat_<float> >& kde_resp_precalc)
{

    int n = dxs.rows;

    cv::Mat_<float> kde_resp;
    float step_size = 0.1;

    // if this has not been precomputer, precompute it, otherwise use it
    if(kde_resp_precalc.find(resp_size) == kde_resp_precalc.end())
    {
        kde_resp = cv::Mat_<float>((int)((resp_size / step_size)*(resp_size/step_size)), resp_size * resp_size);
        cv::MatIterator_<float> kde_it = kde_resp.begin();

        for(int x = 0; x < resp_size/step_size; x++)
        {
            float dx = x * step_size;
            for(int y = 0; y < resp_size/step_size; y++)
            {
                float dy = y * step_size;

                int ii,jj;
                float v,vx,vy;

                for(ii = 0; ii < resp_size; ii++)
                {
                    vx = (dy-ii)*(dy-ii);
                    for(jj = 0; jj < resp_size; jj++)
                    {
                        vy = (dx-jj)*(dx-jj);

                        // the KDE evaluation of that point
                        v = exp(a*(vx+vy));

                        *kde_it++ = v;
                    }
                }
            }
        }

        kde_resp_precalc[resp_size] = kde_resp.clone();
    }
    else
    {
        // use the precomputed version
        kde_resp = kde_resp_precalc.find(resp_size)->second;
    }

    // for every point (patch) calculating mean-shift
    for(int i = 0; i < n; i++)
    {
        if(patch_experts.visibilities[scale][view_id].at<int>(i,0) == 0)
        {
            out_mean_shifts.at<float>(i,0) = 0;
            out_mean_shifts.at<float>(i+n,0) = 0;
            continue;
        }

        // indices of dx, dy
        float dx = dxs.at<float>(i);
        float dy = dys.at<float>(i);

        // Ensure that we are within bounds (important for precalculation)
        if(dx < 0)
            dx = 0;
        if(dy < 0)
            dy = 0;
        if(dx > resp_size - step_size)
            dx = resp_size - step_size;
        if(dy > resp_size - step_size)
            dy = resp_size - step_size;

        // Pick the row from precalculated kde that approximates the current dx, dy best
        int closest_col = (int)(dy /step_size + 0.5); // Plus 0.5 is there, as C++ rounds down with int cast
        int closest_row = (int)(dx /step_size + 0.5); // Plus 0.5 is there, as C++ rounds down with int cast

        int idx = closest_row * ((int)(resp_size/step_size + 0.5)) + closest_col; // Plus 0.5 is there, as C++ rounds down with int cast

        cv::MatIterator_<float> kde_it = kde_resp.begin() + kde_resp.cols*idx;

        float mx=0.0;
        float my=0.0;
        float sum=0.0;

        // Iterate over the patch responses here
        cv::MatConstIterator_<float> p = patch_expert_responses[i].begin();

        for(int ii = 0; ii < resp_size; ii++)
        {
            for(int jj = 0; jj < resp_size; jj++)
            {

                // the KDE evaluation of that point multiplied by the probability at the current, xi, yi
                float v = (*p++) * (*kde_it++);

                sum += v;

                // mean shift in x and y
                mx += v*jj;
                my += v*ii;

            }
        }

        float msx = (mx/sum - dx);
        float msy = (my/sum - dy);

        out_mean_shifts.at<float>(i,0) = msx;
        out_mean_shifts.at<float>(i+n,0) = msy;

    }

}

void CLNF::GetWeightMatrix(cv::Mat_<float>& WeightMatrix, int scale, int view_id, const FaceModelParameters& parameters)
{
    int n = pdm.NumberOfPoints();

    // Is the weight matrix needed at all
    if(parameters.weight_factor > 0)
    {
        WeightMatrix = cv::Mat_<float>::zeros(n*2, n*2);

        for (int p=0; p < n; p++)
        {
            if(!patch_experts.ccnf_expert_intensity.empty())
            {

                // for the x dimension
                WeightMatrix.at<float>(p,p) = WeightMatrix.at<float>(p,p)  + patch_experts.ccnf_expert_intensity[scale][view_id][p].patch_confidence;

                // for they y dimension
                WeightMatrix.at<float>(p+n,p+n) = WeightMatrix.at<float>(p,p);

            }
            else
            {
                // Across the modalities add the confidences
                for(size_t pc=0; pc < patch_experts.svr_expert_intensity[scale][view_id][p].svr_patch_experts.size(); pc++)
                {
                    // for the x dimension
                    WeightMatrix.at<float>(p,p) = WeightMatrix.at<float>(p,p)  + patch_experts.svr_expert_intensity[scale][view_id][p].svr_patch_experts.at(pc).confidence;
                }
                // for the y dimension
                WeightMatrix.at<float>(p+n,p+n) = WeightMatrix.at<float>(p,p);
            }
        }
        WeightMatrix = parameters.weight_factor * WeightMatrix;
    }
    else
    {
        WeightMatrix = cv::Mat_<float>::eye(n*2, n*2);
    }

}

//=============================================================================
double CLNF::NU_RLMS(cv::Vec6d& final_global, cv::Mat_<double>& final_local, const std::vector<cv::Mat_<float> >& patch_expert_responses, const cv::Vec6d& initial_global, const cv::Mat_<double>& initial_local,
                     const cv::Mat_<double>& base_shape, const cv::Matx22d& sim_img_to_ref, const cv::Matx22f& sim_ref_to_img, int resp_size, int view_id, bool rigid, int scale, cv::Mat_<double>& landmark_lhoods,
                     const FaceModelParameters& parameters)
{		

    int n = pdm.NumberOfPoints();

    // Mean, eigenvalues, eigenvectors
    cv::Mat_<double> M = this->pdm.mean_shape;
    cv::Mat_<double> E = this->pdm.eigen_values;
    //Mat_<double> V = this->pdm.princ_comp;

    int m = pdm.NumberOfModes();

    cv::Vec6d current_global(initial_global);

    cv::Mat_<float> current_local;
    initial_local.convertTo(current_local, CV_32F);

    cv::Mat_<double> current_shape;
    cv::Mat_<double> previous_shape;

    // Pre-calculate the regularisation term
    cv::Mat_<float> regTerm;

    if(rigid)
    {
        regTerm = cv::Mat_<float>::zeros(6,6);
    }
    else
    {
        cv::Mat_<double> regularisations = cv::Mat_<double>::zeros(1, 6 + m);

        // Setting the regularisation to the inverse of eigenvalues
        cv::Mat(parameters.reg_factor / E).copyTo(regularisations(cv::Rect(6, 0, m, 1)));
        cv::Mat_<double> regTerm_d = cv::Mat::diag(regularisations.t());
        regTerm_d.convertTo(regTerm, CV_32F);
    }

    cv::Mat_<float> WeightMatrix;
    GetWeightMatrix(WeightMatrix, scale, view_id, parameters);

    cv::Mat_<float> dxs, dys;

    // The preallocated memory for the mean shifts
    cv::Mat_<float> mean_shifts(2 * pdm.NumberOfPoints(), 1, 0.0);

    // Number of iterations
    for(int iter = 0; iter < parameters.num_optimisation_iteration; iter++)
    {
        // get the current estimates of x
        pdm.CalcShape2D(current_shape, current_local, current_global);

        if(iter > 0)
        {
            // if the shape hasn't changed terminate
            if(norm(current_shape, previous_shape) < 0.01)
            {
                break;
            }
        }

        current_shape.copyTo(previous_shape);

        // Jacobian, and transposed weighted jacobian
        cv::Mat_<float> J, J_w_t;

        // calculate the appropriate Jacobians in 2D, even though the actual behaviour is in 3D, using small angle approximation and oriented shape
        if(rigid)
        {
            pdm.ComputeRigidJacobian(current_local, current_global, J, WeightMatrix, J_w_t);
        }
        else
        {
            pdm.ComputeJacobian(current_local, current_global, J, WeightMatrix, J_w_t);
        }

        // useful for mean shift calculation
        float a = -0.5/(parameters.sigma * parameters.sigma);

        cv::Mat_<double> current_shape_2D = current_shape.reshape(1, 2).t();
        cv::Mat_<double> base_shape_2D = base_shape.reshape(1, 2).t();

        cv::Mat_<float> offsets;
        cv::Mat((current_shape_2D - base_shape_2D) * cv::Mat(sim_img_to_ref).t()).convertTo(offsets, CV_32F);

        dxs = offsets.col(0) + (resp_size-1)/2;
        dys = offsets.col(1) + (resp_size-1)/2;

        NonVectorisedMeanShift_precalc_kde(mean_shifts, patch_expert_responses, dxs, dys, resp_size, a, scale, view_id, kde_resp_precalc);

        // Now transform the mean shifts to the the image reference frame, as opposed to one of ref shape (object space)
        cv::Mat_<float> mean_shifts_2D = (mean_shifts.reshape(1, 2)).t();

        mean_shifts_2D = mean_shifts_2D * cv::Mat(sim_ref_to_img).t();
        mean_shifts = cv::Mat(mean_shifts_2D.t()).reshape(1, n*2);

        // remove non-visible observations
        for(int i = 0; i < n; ++i)
        {
            // if patch unavailable for current index
            if(patch_experts.visibilities[scale][view_id].at<int>(i,0) == 0)
            {
                cv::Mat Jx = J.row(i);
                Jx = cvScalar(0);
                cv::Mat Jy = J.row(i+n);
                Jy = cvScalar(0);
                mean_shifts.at<float>(i,0) = 0.0f;
                mean_shifts.at<float>(i+n,0) = 0.0f;
            }
        }

        // projection of the meanshifts onto the jacobians (using the weighted Jacobian, see Baltrusaitis 2013)
        cv::Mat_<float> J_w_t_m = J_w_t * mean_shifts;

        // Add the regularisation term
        if(!rigid)
        {
            J_w_t_m(cv::Rect(0,6,1, m)) = J_w_t_m(cv::Rect(0,6,1, m)) - regTerm(cv::Rect(6,6, m, m)) * current_local;
        }

        // Calculating the Hessian approximation
        cv::Mat_<float> Hessian = J_w_t * J;

        // Add the Tikhonov regularisation
        Hessian = Hessian + regTerm;

        // Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
        cv::Mat_<float> param_update;
        cv::solve(Hessian, J_w_t_m, param_update, CV_CHOLESKY);

        // update the reference
        pdm.UpdateModelParameters(param_update, current_local, current_global);

        // clamp to the local parameters for valid expressions
        pdm.Clamp(current_local, current_global, parameters);

    }

    // compute the log likelihood
    double loglhood = 0;

    landmark_lhoods = cv::Mat_<double>(n, 1, -1e8);

    for(int i = 0; i < n; i++)
    {
        /// whateverx fix it
        /// https://github.com/FaceAR/OpenFaceIOS/issues/1
        if(patch_experts.visibilities[scale][view_id].at<int>(i,0) == 0 || dxs.dims == 0 )
        {
            continue;
        }
        float dx = dxs.at<float>(i);
        float dy = dys.at<float>(i);

        int ii,jj;
        float v,vx,vy,sum=0.0;

        // Iterate over the patch responses here
        cv::MatConstIterator_<float> p = patch_expert_responses[i].begin();

        for(ii = 0; ii < resp_size; ii++)
        {
            vx = (dy-ii)*(dy-ii);
            for(jj = 0; jj < resp_size; jj++)
            {
                vy = (dx-jj)*(dx-jj);

                // the probability at the current, xi, yi
                v = *p++;

                // the KDE evaluation of that point
                v *= exp(-0.5*(vx+vy)/(parameters.sigma * parameters.sigma));

                sum += v;
            }
        }
        landmark_lhoods.at<double>(i,0) = (double)sum;

        // the offset is there for numerical stability
        loglhood += log(sum + 1e-8);

    }
    loglhood = loglhood/sum(patch_experts.visibilities[scale][view_id])[0];

    final_global = current_global;
    final_local = current_local;

    return loglhood;

}


bool CLNF::RemoveBackground(cv::Mat_<float>& out_depth_image, const cv::Mat_<float>& depth_image)
{
    // use the current estimate of the face location to determine what is foreground and background
    double tx = this->params_global[4];
    double ty = this->params_global[5];

    // if we are too close to the edge fail
    if(tx - 9 <= 0 || ty - 9 <= 0 || tx + 9 >= depth_image.cols || ty + 9 >= depth_image.rows)
    {
        std::cout << "Face estimate is too close to the edge, tracking failed" << std::endl;
        return false;
    }

    cv::Mat_<double> current_shape;

    pdm.CalcShape2D(current_shape, params_local, params_global);

    double min_x, max_x, min_y, max_y;

    int n = this->pdm.NumberOfPoints();

    cv::minMaxLoc(current_shape(cv::Range(0, n), cv::Range(0,1)), &min_x, &max_x);
    cv::minMaxLoc(current_shape(cv::Range(n, n*2), cv::Range(0,1)), &min_y, &max_y);

    // the area of interest: size of face with some scaling ( these scalings are fairly ad-hoc)
    double width = 3 * (max_x - min_x);
    double height = 2.5 * (max_y - min_y);

    // getting the region of interest from the depth image,
    // so we don't get other objects lying at same depth as head in the image but away from it
    cv::Rect_<int> roi((int)(tx-width/2), (int)(ty - height/2), (int)width, (int)height);

    // clamp it if it does not lie fully in the image
    if(roi.x < 0) roi.x = 0;
    if(roi.y < 0) roi.y = 0;
    if(roi.width + roi.x >= depth_image.cols) roi.x = depth_image.cols - roi.width;
    if(roi.height + roi.y >= depth_image.rows) roi.y = depth_image.rows - roi.height;

    if(width > depth_image.cols)
    {
        roi.x = 0; roi.width = depth_image.cols;
    }
    if(height > depth_image.rows)
    {
        roi.y = 0; roi.height = depth_image.rows;
    }

    if(roi.width == 0) roi.width = depth_image.cols;
    if(roi.height == 0) roi.height = depth_image.rows;

    if(roi.x >= depth_image.cols) roi.x = 0;
    if(roi.y >= depth_image.rows) roi.y = 0;

    // Initialise the mask
    cv::Mat_<uchar> mask(depth_image.rows, depth_image.cols, (uchar)0);

    cv::Mat_<uchar> valid_pixels = depth_image > 0;

    // check if there is any depth near the estimate
    if(cv::sum(valid_pixels(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16))/255)[0] > 0)
    {
        double Z = cv::mean(depth_image(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16)), valid_pixels(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16)))[0]; // Z offset from the surface of the face

        // Only operate within region of interest of the depth image
        cv::Mat dRoi = depth_image(roi);

        cv::Mat mRoi = mask(roi);

        // Filter all pixels further than 20cm away from the current pose depth estimate
        cv::inRange(dRoi, Z - 200, Z + 200, mRoi);

        // Convert to be either 0 or 1
        mask = mask / 255;

        cv::Mat_<float> maskF;
        mask.convertTo(maskF, CV_32F);

        //Filter the depth image
        out_depth_image = depth_image.mul(maskF);
    }
    else
    {
        std::cout << "No depth signal found in foreground, tracking failed" << std::endl;
        return false;
    }
    return true;
}

// Getting a 3D shape model from the current detected landmarks (in camera space)
cv::Mat_<double> CLNF::GetShape(double fx, double fy, double cx, double cy) const
{
    int n = this->detected_landmarks.rows/2;

    cv::Mat_<double> shape3d(n*3, 1);

    this->pdm.CalcShape3D(shape3d, this->params_local);

    // Need to rotate the shape to get the actual 3D representation

    // get the rotation matrix from the euler angles
    cv::Matx33d R = LandmarkDetector::Euler2RotationMatrix(cv::Vec3d(params_global[1], params_global[2], params_global[3]));

    shape3d = shape3d.reshape(1, 3);

    shape3d = shape3d.t() * cv::Mat(R).t();

    // from the weak perspective model can determine the average depth of the object
    double Zavg = fx / params_global[0];

    cv::Mat_<double> outShape(n,3,0.0);

    // this is described in the paper in section 3.4 (equation 10) (of the CLM-Z paper)
    for(int i = 0; i < n; i++)
    {
        double Z = Zavg + shape3d.at<double>(i,2);

        double X = Z * ((this->detected_landmarks.at<double>(i) - cx)/fx);
        double Y = Z * ((this->detected_landmarks.at<double>(i + n) - cy)/fy);

        outShape.at<double>(i,0) = (double)X;
        outShape.at<double>(i,1) = (double)Y;
        outShape.at<double>(i,2) = (double)Z;

    }

    // The format is 3 rows - n cols
    return outShape.t();

}

// A utility bounding box function
cv::Rect_<double> CLNF::GetBoundingBox() const
{
    cv::Mat_<double> xs = this->detected_landmarks(cv::Rect(0,0,1,this->detected_landmarks.rows/2));
    cv::Mat_<double> ys = this->detected_landmarks(cv::Rect(0,this->detected_landmarks.rows/2, 1, this->detected_landmarks.rows/2));

    double min_x, max_x;
    double min_y, max_y;
    cv::minMaxLoc(xs, &min_x, &max_x);
    cv::minMaxLoc(ys, &min_y, &max_y);

    // See if the detections intersect
    cv::Rect_<double> model_rect(min_x, min_y, max_x - min_x, max_y - min_y);
    return model_rect;
}

// Legacy function not used at the moment
void CLNF::NonVectorisedMeanShift(cv::Mat_<double>& out_mean_shifts, const std::vector<cv::Mat_<float> >& patch_expert_responses, const cv::Mat_<double> &dxs, const cv::Mat_<double> &dys, int resp_size, double a, int scale, int view_id)
{

    int n = dxs.rows;

    for(int i = 0; i < n; i++)
    {

        if(patch_experts.visibilities[scale][view_id].at<int>(i,0) == 0  || sum(patch_expert_responses[i])[0] == 0)
        {
            out_mean_shifts.at<double>(i,0) = 0;
            out_mean_shifts.at<double>(i+n,0) = 0;
            continue;
        }

        // indices of dx, dy
        double dx = dxs.at<double>(i);
        double dy = dys.at<double>(i);

        int ii,jj;
        double v,vx,vy,mx=0.0,my=0.0,sum=0.0;

        // Iterate over the patch responses here
        cv::MatConstIterator_<float> p = patch_expert_responses[i].begin();

        for(ii = 0; ii < resp_size; ii++)
        {
            vx = (dy-ii)*(dy-ii);
            for(jj = 0; jj < resp_size; jj++)
            {
                vy = (dx-jj)*(dx-jj);

                // the probability at the current, xi, yi
                v = *p++;

                // the KDE evaluation of that point
                double kd = exp(a*(vx+vy));
                v *= kd;

                sum += v;

                // mean shift in x and y
                mx += v*jj;
                my += v*ii;

            }
        }

        // setting the actual mean shift update
        double msx = (mx/sum - dx);
        double msy = (my/sum - dy);

        out_mean_shifts.at<double>(i, 0) = msx;
        out_mean_shifts.at<double>(i + n, 0) = msy;

    }
}

