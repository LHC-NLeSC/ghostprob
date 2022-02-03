#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "util.h"

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

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} ghost_logger;

template <typename T>
using UniquePtr = std::unique_ptr<T, util::InferDeleter>;

class GhostDetection
{
    public:
    
        GhostDetection(const std::string& engineFilename);

        bool build();

        bool initialize(const std::string& rootFile);

        bool infer();

    private:

        std::string mEngineFilename;

        UniquePtr<nvinfer1::ICudaEngine> mEngine;

        UniquePtr<nvinfer1::IExecutionContext> mContext;

        const int32_t mInputSize, mOutputSize;

        void* mInputBufferHost;
        void* mInputBufferDevice;
        void* mOutputBufferHost;
        void* mOutputBufferDevice;

        TFile* mRootFile;
        TTree* mEventTree;
};

GhostDetection::GhostDetection(const std::string& engineFilename): mEngineFilename(engineFilename), mEngine(nullptr), 
mContext(nullptr), mInputSize(16),mOutputSize(1),mInputBufferHost(nullptr), mInputBufferDevice(nullptr), 
mOutputBufferHost(nullptr), mOutputBufferDevice(nullptr), mRootFile(nullptr), mEventTree(nullptr){}

bool GhostDetection::initialize(const std::string& rootFile)
{
    // Allocate host and device buffers
    auto inputSize = mInputSize * sizeof(float);
    if(mInputBufferHost != nullptr)
    {
        cudaFreeHost(mInputBufferHost);
    }
    cudaHostalloc(&mInputBufferHost, inputSize, cudaHostAllocDefault);
    if(mInputBufferDevice != nullptr)
    {
        cudaFree(nInputBufferDevice)
    }
    cudaMalloc(&inputBufferDevice, inputSize);

    auto outputSize = mOutputSize * sizeof(float);
    if(mOutputBufferHost != nullptr)
    {
        cudaFreeHost(mOutputBufferHost);
    }
    cudaHostalloc(&mOutputBufferHost, outputSize, cudaHostAllocDefault);
    if(mOutputBufferDevice != nullptr)
    {
        cudaFree(nOutputBufferDevice)
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
    mRootFile = new TFile(rootFile);
    // TODO: No hardcoded TTree name please
    mEventTree = (TTree*)mRootFile->Get("kalman_validator/kalman_ip_tree");
    auto host_vars = static_cast<float*>(mInputBufferHost);
    tree->SetBranchAddress("x", host_vars[0]);
    tree->SetBranchAddress("y", host_vars[1]);
    tree->SetBranchAddress("tx", host_vars[2]);
    tree->SetBranchAddress("ty", host_vars[3]);
    tree->SetBranchAddress("best_qop", host_vars[4]);
    tree->SetBranchAddress("best_pt", host_vars[5]);
    tree->SetBranchAddress("kalman_ip_chi2", host_vars[6]);
    tree->SetBranchAddress("kalman_docaz", host_vars[7]);
    tree->SetBranchAddress("chi2", host_vars[8]);
    tree->SetBranchAddress("chi2V", host_vars[9]);
    tree->SetBranchAddress("chi2UT", host_vars[10]);
    tree->SetBranchAddress("chi2T", host_vars[11]);
    tree->SetBranchAddress("ndof", host_vars[12]);
    tree->SetBranchAddress("ndofV", host_vars[13]);
    tree->SetBranchAddress("ndofT", host_vars[14]);
    tree->SetBranchAddress("nUT", host_vars[15]);

    host_vars = static_cast<float*>(mOutputBufferHost);
    tree->SetBranchAddress("ghost", host_vars[0]);
    
    // Create TensorRT context
    mContext = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext)
    {
        return false;
    }
    auto input_idx = mEngine->getBindingIndex("dense_input");
    auto input_dims = nvinfer1::Dims4{mInputSize};
    mContext->setBindingDimensions(input_idx, input_dims);
}

bool GhostDetection::build()
{
    // Create Builder
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(ghost_logger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    // Create Network
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    // Create config
    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    config->setMaxWorkspaceSize(1024_MiB);

    // Add profile stream to config
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> profileStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        profileStream.reset(nullptr);
    }
    config->setProfileStream(*profileStream);

    // Do the actual building of the network
    UniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    // Create inference runtime
    UniquePtr<IRuntime> runtime{createInferRuntime(ghost_logger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    // Create engine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
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

    // Memcpy from host input buffers to device input buffers
    cudaMemcpy(mInputBufferHost, mInputBufferDevice, mInputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Do inference, batch size 1
    bool status = mContext->executeV2(mInputBufferDevice);
    if (!status)
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    cudaMemcpy(mOutputBufferDevice, mOutputBufferHost, mOutputSize * sizeof(float), cudaMemcpyDeviceToHost);

    return true;
}
