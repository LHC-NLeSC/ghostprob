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
//#include "util.h"

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
        if (severity <= nvinfer1::ILogger::Severity::kINFO)
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

        std::unique_ptr<nvinfer1::ICudaEngine> mEngine;

        std::unique_ptr<nvinfer1::IExecutionContext> mContext;

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
    cudaHostAlloc(&mInputBufferHost, inputSize, cudaHostAllocDefault);
    if(mInputBufferDevice != nullptr)
    {
        cudaFree(mInputBufferDevice);
    }
    cudaMalloc(&mInputBufferDevice, inputSize);

    auto outputSize = mOutputSize * sizeof(int32_t);
    if(mOutputBufferHost != nullptr)
    {
        cudaFreeHost(mOutputBufferHost);
    }
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
    mEventTree->SetBranchAddress("x", &host_vars);
    mEventTree->SetBranchAddress("y", &host_vars[1]);
    mEventTree->SetBranchAddress("tx", &host_vars[2]);
    mEventTree->SetBranchAddress("ty", &host_vars[3]);
    mEventTree->SetBranchAddress("best_qop", &host_vars[4]);
    mEventTree->SetBranchAddress("best_pt", &host_vars[5]);
    mEventTree->SetBranchAddress("kalman_ip_chi2", &host_vars[6]);
    mEventTree->SetBranchAddress("kalman_docaz", &host_vars[7]);
    mEventTree->SetBranchAddress("chi2", &host_vars[8]);
    mEventTree->SetBranchAddress("chi2V", &host_vars[9]);
    mEventTree->SetBranchAddress("chi2UT", &host_vars[10]);
    mEventTree->SetBranchAddress("chi2T", &host_vars[11]);
    mEventTree->SetBranchAddress("ndof", &host_vars[12]);
    mEventTree->SetBranchAddress("ndofV", &host_vars[13]);
    mEventTree->SetBranchAddress("ndofT", &host_vars[14]);
    mEventTree->SetBranchAddress("nUT", &host_vars[15]);

    mEventTree->SetBranchAddress("ghost", static_cast<int32_t*>(mOutputBufferHost));
    
    // Create TensorRT context
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext)
    {
        return false;
    }
    auto input_idx = mEngine->getBindingIndex("dense_input");
    auto input_dims = nvinfer1::Dims{mInputSize};
    mContext->setBindingDimensions(input_idx, input_dims);
}

bool GhostDetection::build()
{
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(ghost_logger));
    if (!builder)
    {
        return false;
    }

    // Create (empty) network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    // Create config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    config->setMaxWorkspaceSize(1024 * (1 << 20));

    // Add profile stream to config
/*    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> profileStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        profileStream.reset(nullptr);
    }
    config->setProfileStream(*profileStream);*/

    // Define the ONNX parser with the network it should write to
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, ghost_logger));
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
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    // Create inference runtime
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(ghost_logger)};
    if (!runtime)
    {
        return false;
    }

    // Create engine
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
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
    bool status = mContext->executeV2((void* const*)mInputBufferDevice);
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
    ghostinfer.initialize("PrCheckerPlots.root");
    for(int32_t i = 0; i < 1000; ++i)
    {
        ghostinfer.infer(i);
    }
}
