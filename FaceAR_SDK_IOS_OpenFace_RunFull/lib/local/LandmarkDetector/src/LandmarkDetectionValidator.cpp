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

#include "LandmarkDetectionValidator.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// System includes
#include <fstream>

// Math includes
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Local includes
#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;

// Copy constructor
DetectionValidator::DetectionValidator(const DetectionValidator& other) : orientations(other.orientations), bs(other.bs), paws(other.paws),
    cnn_subsampling_layers(other.cnn_subsampling_layers), cnn_layer_types(other.cnn_layer_types), cnn_fully_connected_layers_bias(other.cnn_fully_connected_layers_bias),
    cnn_convolutional_layers_bias(other.cnn_convolutional_layers_bias), cnn_convolutional_layers_dft(other.cnn_convolutional_layers_dft)
{

    this->validator_type = other.validator_type;

    this->activation_fun = other.activation_fun;
    this->output_fun = other.output_fun;

    this->ws.resize(other.ws.size());
    for (size_t i = 0; i < other.ws.size(); ++i)
    {
        // Make sure the matrix is copied.
        this->ws[i] = other.ws[i].clone();
    }

    this->ws_nn.resize(other.ws_nn.size());
    for (size_t i = 0; i < other.ws_nn.size(); ++i)
    {
        this->ws_nn[i].resize(other.ws_nn[i].size());

        for (size_t k = 0; k < other.ws_nn[i].size(); ++k)
        {
            // Make sure the matrix is copied.
            this->ws_nn[i][k] = other.ws_nn[i][k].clone();
        }
    }

    this->cnn_convolutional_layers.resize(other.cnn_convolutional_layers.size());
    for (size_t v = 0; v < other.cnn_convolutional_layers.size(); ++v)
    {
        this->cnn_convolutional_layers[v].resize(other.cnn_convolutional_layers[v].size());

        for (size_t l = 0; l < other.cnn_convolutional_layers[v].size(); ++l)
        {
            this->cnn_convolutional_layers[v][l].resize(other.cnn_convolutional_layers[v][l].size());

            for (size_t i = 0; i < other.cnn_convolutional_layers[v][l].size(); ++i)
            {
                this->cnn_convolutional_layers[v][l][i].resize(other.cnn_convolutional_layers[v][l][i].size());

                for (size_t k = 0; k < other.cnn_convolutional_layers[v][l][i].size(); ++k)
                {
                    // Make sure the matrix is copied.
                    this->cnn_convolutional_layers[v][l][i][k] = other.cnn_convolutional_layers[v][l][i][k].clone();
                }

            }
        }
    }

    this->cnn_fully_connected_layers.resize(other.cnn_fully_connected_layers.size());
    for (size_t v = 0; v < other.cnn_fully_connected_layers.size(); ++v)
    {
        this->cnn_fully_connected_layers[v].resize(other.cnn_fully_connected_layers[v].size());

        for (size_t l = 0; l < other.cnn_fully_connected_layers[v].size(); ++l)
        {
            // Make sure the matrix is copied.
            this->cnn_fully_connected_layers[v][l] = other.cnn_fully_connected_layers[v][l].clone();
        }
    }

    this->mean_images.resize(other.mean_images.size());
    for (size_t i = 0; i < other.mean_images.size(); ++i)
    {
        // Make sure the matrix is copied.
        this->mean_images[i] = other.mean_images[i].clone();
    }

    this->standard_deviations.resize(other.standard_deviations.size());
    for (size_t i = 0; i < other.standard_deviations.size(); ++i)
    {
        // Make sure the matrix is copied.
        this->standard_deviations[i] = other.standard_deviations[i].clone();
    }

}

//===========================================================================
// Read in the landmark detection validation module
void DetectionValidator::Read(std::string location)
{

    std::ifstream detection_validator_stream (location, std::ios::in | std::ios::binary);
    if (detection_validator_stream.is_open())
    {
        detection_validator_stream.seekg (0, std::ios::beg);

        // Read validator type
        detection_validator_stream.read ((char*)&validator_type, 4);

        // Read the number of views (orientations) within the validator
        int n;
        detection_validator_stream.read ((char*)&n, 4);

        orientations.resize(n);

        for(int i = 0; i < n; i++)
        {
            cv::Mat_<double> orientation_tmp;
            LandmarkDetector::ReadMatBin(detection_validator_stream, orientation_tmp);

            orientations[i] = cv::Vec3d(orientation_tmp.at<double>(0), orientation_tmp.at<double>(1), orientation_tmp.at<double>(2));

            // Convert from degrees to radians
            orientations[i] = orientations[i] * M_PI / 180.0;
        }

        // Initialise the piece-wise affine warps, biases and weights
        paws.resize(n);

        if( validator_type == 0)
        {
            // Reading in SVRs
            bs.resize(n);
            ws.resize(n);
        }
        else if(validator_type == 1)
        {
            // Reading in NNs
            ws_nn.resize(n);

            activation_fun.resize(n);
            output_fun.resize(n);
        }
        else if(validator_type == 2)
        {
            cnn_convolutional_layers.resize(n);
            cnn_convolutional_layers_dft.resize(n);
            cnn_subsampling_layers.resize(n);
            cnn_fully_connected_layers.resize(n);
            cnn_layer_types.resize(n);
            cnn_fully_connected_layers_bias.resize(n);
            cnn_convolutional_layers_bias.resize(n);
        }

        // Initialise the normalisation terms
        mean_images.resize(n);
        standard_deviations.resize(n);

        // Read in the validators for each of the views
        for(int i = 0; i < n; i++)
        {

            // Read in the mean images
            LandmarkDetector::ReadMatBin(detection_validator_stream, mean_images[i]);
            mean_images[i] = mean_images[i].t();

            LandmarkDetector::ReadMatBin(detection_validator_stream, standard_deviations[i]);
            standard_deviations[i] = standard_deviations[i].t();

            // Model specifics
            if(validator_type == 0)
            {
                // Reading in the biases and weights
                detection_validator_stream.read ((char*)&bs[i], 8);
                LandmarkDetector::ReadMatBin(detection_validator_stream, ws[i]);

            }
            else if(validator_type == 1)
            {

                // Reading in the number of layers in the neural net
                int num_depth_layers;
                detection_validator_stream.read ((char*)&num_depth_layers, 4);

                // Reading in activation and output function types
                detection_validator_stream.read ((char*)&activation_fun[i], 4);
                detection_validator_stream.read ((char*)&output_fun[i], 4);

                ws_nn[i].resize(num_depth_layers);
                for(int layer = 0; layer < num_depth_layers; layer++)
                {
                    LandmarkDetector::ReadMatBin(detection_validator_stream, ws_nn[i][layer]);

                    // Transpose for efficiency during multiplication
                    ws_nn[i][layer] = ws_nn[i][layer].t();
                }
            }
            else if(validator_type == 2)
            {
                // Reading in CNNs

                int network_depth;
                detection_validator_stream.read ((char*)&network_depth, 4);

                cnn_layer_types[i].resize(network_depth);

                for(int layer = 0; layer < network_depth; ++layer)
                {

                    int layer_type;
                    detection_validator_stream.read ((char*)&layer_type, 4);
                    cnn_layer_types[i][layer] = layer_type;

                    // convolutional
                    if(layer_type == 0)
                    {

                        // Read the number of input maps
                        int num_in_maps;
                        detection_validator_stream.read ((char*)&num_in_maps, 4);

                        // Read the number of kernels for each input map
                        int num_kernels;
                        detection_validator_stream.read ((char*)&num_kernels, 4);

                        std::vector<std::vector<cv::Mat_<float> > > kernels;
                        std::vector<std::vector<std::pair<int, cv::Mat_<double> > > > kernel_dfts;

                        kernels.resize(num_in_maps);
                        kernel_dfts.resize(num_in_maps);

                        std::vector<float> biases;
                        for (int k = 0; k < num_kernels; ++k)
                        {
                            float bias;
                            detection_validator_stream.read ((char*)&bias, 4);
                            biases.push_back(bias);
                        }

                        cnn_convolutional_layers_bias[i].push_back(biases);

                        // For every input map
                        for (int in = 0; in < num_in_maps; ++in)
                        {
                            kernels[in].resize(num_kernels);
                            kernel_dfts[in].resize(num_kernels);

                            // For every kernel on that input map
                            for (int k = 0; k < num_kernels; ++k)
                            {
                                ReadMatBin(detection_validator_stream, kernels[in][k]);

                                // Flip the kernel in order to do convolution and not correlation
                                cv::flip(kernels[in][k], kernels[in][k], -1);
                            }
                        }

                        cnn_convolutional_layers[i].push_back(kernels);
                        cnn_convolutional_layers_dft[i].push_back(kernel_dfts);
                    }
                    else if(layer_type == 1)
                    {
                        // Subsampling layer

                        int scale;
                        detection_validator_stream.read ((char*)&scale, 4);

                        cnn_subsampling_layers[i].push_back(scale);
                    }
                    else if(layer_type == 2)
                    {
                        float bias;
                        detection_validator_stream.read ((char*)&bias, 4);
                        cnn_fully_connected_layers_bias[i].push_back(bias);

                        // Fully connected layer
                        cv::Mat_<float> weights;
                        ReadMatBin(detection_validator_stream, weights);
                        cnn_fully_connected_layers[i].push_back(weights);
                    }
                }
            }

            // Read in the piece-wise affine warps
            paws[i].Read(detection_validator_stream);
        }

    }
    else
    {
        std::cout << "WARNING: Can't find the Face checker location" << std::endl;
    }
}

//===========================================================================
// Check if the fitting actually succeeded
double DetectionValidator::Check(const cv::Vec3d& orientation, const cv::Mat_<uchar>& intensity_img, cv::Mat_<double>& detected_landmarks)
{

    int id = GetViewId(orientation);

    // The warped (cropped) image, corresponding to a face lying withing the detected lanmarks
    cv::Mat_<double> warped;

    // the piece-wise affine image
    cv::Mat_<double> intensity_img_double;
    intensity_img.convertTo(intensity_img_double, CV_64F);

    paws[id].Warp(intensity_img_double, warped, detected_landmarks);

    double dec;
    if(validator_type == 0)
    {
        dec = CheckSVR(warped, id);
    }
    else if(validator_type == 1)
    {
        dec = CheckNN(warped, id);
    }
    else if(validator_type == 2)
    {
        dec = CheckCNN(warped, id);
    }
    return dec;
}

double DetectionValidator::CheckNN(const cv::Mat_<double>& warped_img, int view_id)
{
    cv::Mat_<double> feature_vec;
    NormaliseWarpedToVector(warped_img, feature_vec, view_id);
    feature_vec = feature_vec.t();

    for(size_t layer = 0; layer < ws_nn[view_id].size(); ++layer)
    {
        // Add a bias term
        cv::hconcat(cv::Mat_<double>(1,1, 1.0), feature_vec, feature_vec);

        // Apply the weights
        feature_vec = feature_vec * ws_nn[view_id][layer];

        // Activation or output
        int fun_type;
        if(layer != ws_nn[view_id].size() - 1)
        {
            fun_type = activation_fun[view_id];
        }
        else
        {
            fun_type = output_fun[view_id];
        }

        if(fun_type == 0)
        {
            cv::exp(-feature_vec, feature_vec);
            feature_vec = 1.0 /(1.0 + feature_vec);
        }
        else if(fun_type == 1)
        {
            cv::MatIterator_<double> q1 = feature_vec.begin(); // respone for each pixel
            cv::MatIterator_<double> q2 = feature_vec.end();

            // the logistic function (sigmoid) applied to the response
            while(q1 != q2)
            {
                *q1 = 1.7159 * tanh((2.0/3.0) * (*q1));
                q1++;
            }
        }
        // TODO ReLU

    }

    // Turn it to -1, 1 range
    double dec = (feature_vec.at<double>(0) - 0.5) * 2;

    return dec;

}

double DetectionValidator::CheckSVR(const cv::Mat_<double>& warped_img, int view_id)
{

    cv::Mat_<double> feature_vec;
    NormaliseWarpedToVector(warped_img, feature_vec, view_id);


    double dec = (ws[view_id].dot(feature_vec.t()) + bs[view_id]);

    return dec;

}

// Convolutional Neural Network
double DetectionValidator::CheckCNN(const cv::Mat_<double>& warped_img, int view_id)
{

    cv::Mat_<double> feature_vec;
    NormaliseWarpedToVector(warped_img, feature_vec, view_id);

    // Create a normalised image from the crop vector
    cv::Mat_<float> img(warped_img.size(), 0.0);
    img = img.t();

    cv::Mat mask = paws[view_id].pixel_mask.t();
    cv::MatIterator_<uchar>  mask_it = mask.begin<uchar>();

    cv::MatIterator_<double> feature_it = feature_vec.begin();
    cv::MatIterator_<float> img_it = img.begin();

    int wInt = img.cols;
    int hInt = img.rows;

    for(int i=0; i < wInt; ++i)
    {
        for(int j=0; j < hInt; ++j, ++mask_it, ++img_it)
        {
            // if is within mask
            if(*mask_it)
            {
                // assign the feature to image if it is within the mask
                *img_it = (float)*feature_it++;
            }
        }
    }
    img = img.t();

    int cnn_layer = 0;
    int subsample_layer = 0;
    int fully_connected_layer = 0;

    std::vector<cv::Mat_<float> > input_maps;
    input_maps.push_back(img);

    std::vector<cv::Mat_<float> > outputs;

    for(size_t layer = 0; layer < cnn_layer_types[view_id].size(); ++layer)
    {
        // Determine layer type
        int layer_type = cnn_layer_types[view_id][layer];


        // Convolutional layer
        if(layer_type == 0)
        {
            std::vector<cv::Mat_<float> > outputs_kern;
            for(size_t in = 0; in < input_maps.size(); ++in)
            {
                cv::Mat_<float> input_image = input_maps[in];

                // Useful precomputed data placeholders for quick correlation (convolution)
                cv::Mat_<double> input_image_dft;
                cv::Mat integral_image;
                cv::Mat integral_image_sq;

                for(size_t k = 0; k < cnn_convolutional_layers[view_id][cnn_layer][in].size(); ++k)
                {
                    cv::Mat_<float> kernel = cnn_convolutional_layers[view_id][cnn_layer][in][k];

                    // The convolution (with precomputation)
                    cv::Mat_<float> output;
                    if(cnn_convolutional_layers_dft[view_id][cnn_layer][in][k].second.empty())
                    {
                        std::map<int, cv::Mat_<double> > precomputed_dft;

                        LandmarkDetector::matchTemplate_m(input_image, input_image_dft, integral_image, integral_image_sq, kernel, precomputed_dft, output, CV_TM_CCORR);

                        cnn_convolutional_layers_dft[view_id][cnn_layer][in][k].first = precomputed_dft.begin()->first;
                        cnn_convolutional_layers_dft[view_id][cnn_layer][in][k].second = precomputed_dft.begin()->second;
                    }
                    else
                    {
                        std::map<int, cv::Mat_<double> > precomputed_dft;
                        precomputed_dft[cnn_convolutional_layers_dft[view_id][cnn_layer][in][k].first] = cnn_convolutional_layers_dft[view_id][cnn_layer][in][k].second;
                        LandmarkDetector::matchTemplate_m(input_image, input_image_dft, integral_image, integral_image_sq, kernel,  precomputed_dft, output, CV_TM_CCORR);
                    }

                    // Combining the maps
                    if(in == 0)
                    {
                        outputs_kern.push_back(output);
                    }
                    else
                    {
                        outputs_kern[k] = outputs_kern[k] + output;
                    }

                }

            }

            outputs.clear();
            for(size_t k = 0; k < cnn_convolutional_layers[view_id][cnn_layer][0].size(); ++k)
            {
                // Apply the sigmoid
                cv::exp(-outputs_kern[k] - cnn_convolutional_layers_bias[view_id][cnn_layer][k], outputs_kern[k]);
                outputs_kern[k] = 1.0 /(1.0 + outputs_kern[k]);

                outputs.push_back(outputs_kern[k]);

            }

            cnn_layer++;
        }
        if(layer_type == 1)
        {
            // Subsampling layer
            int scale = cnn_subsampling_layers[view_id][subsample_layer];

            cv::Mat kx = cv::Mat::ones(2, 1, CV_32F)*1.0f/scale;
            cv::Mat ky = cv::Mat::ones(1, 2, CV_32F)*1.0f/scale;

            std::vector<cv::Mat_<float> > outputs_sub;
            for(size_t in = 0; in < input_maps.size(); ++in)
            {

                cv::Mat_<float> conv_out;

                cv::sepFilter2D(input_maps[in], conv_out, CV_32F, kx, ky);
                conv_out = conv_out(cv::Rect(1, 1, conv_out.cols - 1, conv_out.rows - 1));

                int res_rows = conv_out.rows / scale;
                int res_cols = conv_out.cols / scale;

                if(conv_out.rows % scale != 0)
                {
                    res_rows++;
                }
                if(conv_out.cols % scale != 0)
                {
                    res_cols++;
                }

                cv::Mat_<float> sub_out(res_rows, res_cols);
                for(int w = 0; w < conv_out.cols; w+=scale)
                {
                    for(int h=0; h < conv_out.rows; h+=scale)
                    {
                        sub_out.at<float>(h/scale, w/scale) = conv_out(h, w);
                    }
                }
                outputs_sub.push_back(sub_out);
            }
            outputs = outputs_sub;
            subsample_layer++;

        }
        if(layer_type == 2)
        {
            // Concatenate all the maps
            cv::Mat_<float> input_concat = input_maps[0].t();
            input_concat = input_concat.reshape(0, 1);

            for(size_t in = 1; in < input_maps.size(); ++in)
            {
                cv::Mat_<float> add = input_maps[in].t();
                add = add.reshape(0,1);
                cv::hconcat(input_concat, add, input_concat);
            }

            input_concat = input_concat * cnn_fully_connected_layers[view_id][fully_connected_layer].t();

            cv::exp(-input_concat - cnn_fully_connected_layers_bias[view_id][fully_connected_layer], input_concat);
            input_concat = 1.0 /(1.0 + input_concat);

            outputs.clear();
            outputs.push_back(input_concat);

            fully_connected_layer++;
        }
        // Set the outputs of this layer to inputs of the next
        input_maps = outputs;

    }

    // Turn it to -1, 1 range
    double dec = (outputs[0].at<float>(0) - 0.5) * 2.0;

    return dec;
}

void DetectionValidator::NormaliseWarpedToVector(const cv::Mat_<double>& warped_img, cv::Mat_<double>& feature_vec, int view_id)
{
    cv::Mat_<double> warped_t = warped_img.t();

    // the vector to be filled with paw values
    cv::MatIterator_<double> vp;
    cv::MatIterator_<double>  cp;

    cv::Mat_<double> vec(paws[view_id].number_of_pixels,1);
    vp = vec.begin();

    cp = warped_t.begin();

    int wInt = warped_img.cols;
    int hInt = warped_img.rows;

    // the mask indicating if point is within or outside the face region

    cv::Mat maskT = paws[view_id].pixel_mask.t();

    cv::MatIterator_<uchar>  mp = maskT.begin<uchar>();

    for(int i=0; i < wInt; ++i)
    {
        for(int j=0; j < hInt; ++j, ++mp, ++cp)
        {
            // if is within mask
            if(*mp)
            {
                *vp++ = *cp;
            }
        }
    }

    // Local normalisation
    cv::Scalar mean;
    cv::Scalar std;
    cv::meanStdDev(vec, mean, std);

    // subtract the mean image
    vec -= mean[0];

    // Normalise the image
    if(std[0] == 0)
    {
        std[0] = 1;
    }

    vec /= std[0];

    // Global normalisation
    feature_vec = (vec - mean_images[view_id])  / standard_deviations[view_id];
}

// Getting the closest view center based on orientation
int DetectionValidator::GetViewId(const cv::Vec3d& orientation) const
{
    int id = 0;

    double dbest = -1.0;

    for(size_t i = 0; i < this->orientations.size(); i++)
    {

        // Distance to current view
        double d = cv::norm(orientation, this->orientations[i]);

        if(i == 0 || d < dbest)
        {
            dbest = d;
            id = i;
        }
    }
    return id;

}



