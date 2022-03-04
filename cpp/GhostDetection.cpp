#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntimeCommon.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
//        if (severity <= nvinfer1::ILogger::Severity::kINFO)
            std::cout << msg << std::endl;
    }
} ghost_logger;

class GhostDetection
{
    public:
    
        GhostDetection(const std::string& engineFilename);

        bool build();

        bool initialize(const std::string& rootFile);

        bool infer(const int32_t nevent);

    private:

        std::string mEngineFilename;

        std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> mEngine;

        std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> mContext;
                
        const int32_t mInputSize, mOutputSize;

        void* mInputBufferHost;
        uint32_t*  mInputBufferHostInt;
        void* mInputBufferDevice;
        void* mOutputBufferHost;
        void* mOutputBufferDevice;

        TFile* mRootFile;
        TTree* mEventTree;
};

GhostDetection::GhostDetection(const std::string& engineFilename): mEngineFilename(engineFilename), mEngine(nullptr), 
mContext(nullptr), mInputSize(17),mOutputSize(1), mInputBufferHost(nullptr), mInputBufferHostInt(nullptr), mInputBufferDevice(nullptr), 
mOutputBufferHost(nullptr), mOutputBufferDevice(nullptr), mRootFile(nullptr), mEventTree(nullptr){}

bool GhostDetection::initialize(const std::string& rootFile)
{
    // Allocate host and device buffers
    auto inputSize = mInputSize * sizeof(float);
    if(mInputBufferHost != nullptr)
    {
        cudaFreeHost(mInputBufferHost);
    }
    cudaHostAlloc(&mInputBufferHost, inputSize, cudaHostAllocDefault);
    mInputBufferHostInt = new uint32_t[4];
    if(mInputBufferDevice != nullptr)
    {
        cudaFree(mInputBufferDevice);
    }
    cudaMalloc(&mInputBufferDevice, inputSize);

    if(mOutputBufferHost != nullptr)
    {
        cudaFreeHost(mOutputBufferHost);
    }
    auto outputSize = mOutputSize * sizeof(float);
    cudaHostAlloc(&mOutputBufferHost, outputSize, cudaHostAllocDefault);
    if(mOutputBufferDevice != nullptr)
    {
        cudaFree(mOutputBufferDevice);
    }
    cudaMalloc(&mOutputBufferDevice, outputSize);

    // Open ROOT File
    if(mRootFile != nullptr)
    {
        mRootFile->Close();
        delete mRootFile;
        mRootFile = nullptr;
        mEventTree = nullptr;
    }

    // Create ROOT event tree
    mRootFile = new TFile(rootFile.c_str());
    // TODO: No hardcoded TTree name please
    mEventTree = (TTree*)mRootFile->Get("kalman_validator/kalman_ip_tree");
    auto host_vars = static_cast<float*>(mInputBufferHost);
    mEventTree->SetBranchAddress("x", &host_vars[1]);
    mEventTree->SetBranchAddress("y", &host_vars[2]);
    mEventTree->SetBranchAddress("tx", &host_vars[3]);
    mEventTree->SetBranchAddress("ty", &host_vars[4]);
    mEventTree->SetBranchAddress("best_qop", &host_vars[5]);
    mEventTree->SetBranchAddress("best_pt", &host_vars[6]);
    mEventTree->SetBranchAddress("kalman_ip_chi2", &host_vars[7]);
    mEventTree->SetBranchAddress("kalman_docaz", &host_vars[8]);
    mEventTree->SetBranchAddress("chi2", &host_vars[9]);
    mEventTree->SetBranchAddress("chi2V", &host_vars[10]);
    mEventTree->SetBranchAddress("chi2UT", &host_vars[11]);
    mEventTree->SetBranchAddress("chi2T", &host_vars[12]);

    mEventTree->SetBranchAddress("ndof", &mInputBufferHostInt[0]);
    mEventTree->SetBranchAddress("ndofV", &mInputBufferHostInt[1]);
    mEventTree->SetBranchAddress("ndofT", &mInputBufferHostInt[2]);
    mEventTree->SetBranchAddress("nUT", &mInputBufferHostInt[3]);

    mEventTree->SetBranchAddress("ghost", static_cast<unsigned*>(mOutputBufferHost));
    
    // Create TensorRT context
    mContext = std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter>(mEngine->createExecutionContext());
    if (!mContext)
    {
        return false;
    }
//    auto input_idx  = mEngine->getBindingIndex("dense_input");
//    auto output_idx = mEngine->getBindingIndex("dense");
//    std::cerr<<"INPUT INDEX :"<<input_idx<<std::endl;
//    std::cerr<<"OUTPUT INDEX:"<<output_idx<<std::endl;
//    auto input_dims = nvinfer1::Dims2(1, mInputSize);
//    mContext->setBindingDimensions(input_idx, input_dims);
}

bool GhostDetection::build()
{
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(nvinfer1::createInferBuilder(ghost_logger));
    if (!builder)
    {
        return false;
    }

    // Create (empty) network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    // Create config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    config->setMaxWorkspaceSize(1024 * (1 << 20));

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    auto input_dims = nvinfer1::Dims2(1, mInputSize);
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kOPT, input_dims);
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kMAX, input_dims);

    config->addOptimizationProfile(profile);

    // Add profile stream to config
/*    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> profileStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        profileStream.reset(nullptr);
    }
    config->setProfileStream(*profileStream);*/

    // Define the ONNX parser with the network it should write to
    auto parser = std::unique_ptr<nvonnxparser::IParser, InferDeleter>(nvonnxparser::createParser(*network, ghost_logger));
    if (!parser)
    {
        return false;
    }

    // Parse the ONNX input network file
    auto parsed = parser->parseFromFile(mEngineFilename.c_str(), 0);
    if (!parsed)
    {
        return false;
    }

    // Do the actual building of the network
    std::unique_ptr<nvinfer1::IHostMemory, InferDeleter> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    // Create inference runtime
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime{nvinfer1::createInferRuntime(ghost_logger)};
    if (!runtime)
    {
        return false;
    }

    // Create engine
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine)
    {
        return false;
    }

    return true;
}


bool GhostDetection::infer(int32_t nevent)
{
    // Read next event in ROOT file
    if(mEventTree == nullptr)
    {
        return false;
    }
    mEventTree->GetEntry(nevent);
    
    auto input_vars = (float*)mInputBufferHost;
    input_vars[0] = 1./ std::abs(input_vars[5]);
    input_vars[13] = (float)(mInputBufferHostInt[0]);
    input_vars[14] = (float)(mInputBufferHostInt[1]);
    input_vars[15] = (float)(mInputBufferHostInt[2]);
    input_vars[16] = (float)(mInputBufferHostInt[3]);


    // Memcpy from host input buffers to device input buffers
    cudaMemcpy(mInputBufferHost, mInputBufferDevice, mInputSize * sizeof(float), cudaMemcpyHostToDevice);

    void* buffers[2];
    buffers[0] = mInputBufferDevice;
    buffers[1] = mOutputBufferDevice;

    // Do inference, batch size 1
    bool status = mContext->executeV2((void* const*)buffers);
    if (!status)
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    cudaMemcpy(mOutputBufferDevice, mOutputBufferHost, mOutputSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cerr<<"Inference result for evt "<<nevent<<": "<<*(static_cast<int32_t*>(mOutputBufferHost))<<std::endl;

    return true;
}


int main(int argc, char* argv[])
{
    GhostDetection ghostinfer("../data/ghost_nn.onnx");
    ghostinfer.build();
    ghostinfer.initialize("../data/PrCheckerPlots.root");
    for(int32_t i = 0; i < 10000; ++i)
    {
        ghostinfer.infer(i);
    }
}
