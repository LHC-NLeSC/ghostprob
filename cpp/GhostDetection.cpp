#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <chrono>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntimeCommon.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

// Constants
const bool useFP16 = false;


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

struct InferenceResult
{
    int32_t true_positives;
    int32_t true_negatives;
    int32_t false_positives;
    int32_t false_negatives;

    int32_t n_events()
    {
        return true_positives + true_negatives + false_positives + false_negatives;
    }

    float accuracy()
    {
        return float(true_positives + true_negatives)/(float)n_events();
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

class NetworkInputDescriptor
{
    public:

        virtual ~NetworkInputDescriptor();

        // Branch names + _i or _f for integer/float branches
        virtual std::vector<std::string> get_branch_names() const=0;

        // Size of the host(+device) input buffers needed for the NN
        virtual unsigned host_buffer_size() const=0;

        // Do the magic mapping the root file variables to the NN input
        virtual void fill_host_buffer(const float* root_buffer, float* host_buffer) const=0;
};

NetworkInputDescriptor::~NetworkInputDescriptor(){}

class GhostNetworkInputDescriptor: public NetworkInputDescriptor
{
    public:

        GhostNetworkInputDescriptor();

        ~GhostNetworkInputDescriptor();

        std::vector<std::string> get_branch_names() const;

        unsigned host_buffer_size() const;

        // Do the magic mapping the root file variables to the NN input
        void fill_host_buffer(const float* root_buffer, float* host_buffer) const;
};

GhostNetworkInputDescriptor::GhostNetworkInputDescriptor(){}

GhostNetworkInputDescriptor::~GhostNetworkInputDescriptor(){}

std::vector<std::string> GhostNetworkInputDescriptor::get_branch_names() const
{
    return std::vector<std::string>({"x_f", "y_f", "tx_f", "ty_f", "best_qop_f", "best_pt_f", "kalman_ip_chi2_f", "kalman_docaz_f", "chi2_f", "chi2V_f",
                                    "chi2UT_f", "chi2T_f", "ndof_i", "ndofV_i", "ndofT_i", "nUT_i"});
}

unsigned GhostNetworkInputDescriptor::host_buffer_size() const
{
    return 17 * sizeof(float);
}

void GhostNetworkInputDescriptor::fill_host_buffer(const float* root_buffer, float* host_buffer) const
{
    host_buffer[0] = 1./ std::abs(root_buffer[4]);
    std::memcpy((void*)(host_buffer + 1), (const void*)root_buffer, 12 * sizeof(float));

    host_buffer[13] = (float)(*(int32_t*)(root_buffer + 12));
    host_buffer[14] = (float)(*(int32_t*)(root_buffer + 13));
    host_buffer[15] = (float)(*(int32_t*)(root_buffer + 14));
    host_buffer[16] = (float)(*(int32_t*)(root_buffer + 15));

}

class InputDataProvider
{
    public:

        InputDataProvider(const NetworkInputDescriptor* network);

        virtual ~InputDataProvider();

        virtual bool load(const std::string& rootFile, const std::string& treeName)=0;

        virtual bool read_event(const int32_t event)=0;

        virtual bool fill_device_buffer(void* buffer)=0;

    protected:

        bool open(  const std::string& rootFile, const std::string treeName, const std::vector<std::string>& branchNames,
                    float* inputBuffer, float* outputBuffer);

        bool close();

        const NetworkInputDescriptor* mNetworkDescriptor;
        TFile* mRootFile;
        TTree* mEventTree;
    
};

InputDataProvider::InputDataProvider(const NetworkInputDescriptor* network): mNetworkDescriptor(network), mRootFile(nullptr), mEventTree(nullptr){}

InputDataProvider::~InputDataProvider()
{
    close();
}

bool InputDataProvider::open(const std::string& rootFile, const std::string treeName, const std::vector<std::string>& branchNames, 
                        float* inputBuffer, float* outputBuffer)
{
    close();
    mRootFile = new TFile(rootFile.c_str());
    mEventTree = (TTree*)mRootFile->Get(treeName.c_str());
    for(int32_t i = 0; i < branchNames.size(); ++i)
    {
        auto typechar = branchNames[i].back();
        auto branchName = branchNames[i].substr(0, branchNames[i].size()-2);
        if(typechar == 'i')
        {
            mEventTree->SetBranchAddress(branchName.c_str(), (int32_t*)(&inputBuffer[i]));
        }
        else if(typechar == 'f')
        {
            mEventTree->SetBranchAddress(branchName.c_str(), &inputBuffer[i]);
        }
        else
        {
            throw 1;
        }
    }

    mEventTree->SetBranchAddress("ghost", (unsigned*)outputBuffer);
}

bool InputDataProvider::close()
{
    if(mEventTree != nullptr)
    {
        delete mEventTree;
    }
    if(mRootFile != nullptr)
    {
        mRootFile->Close();
        delete mRootFile;
    }
}

class FileInputDataProvider: public InputDataProvider
{
    public:

        FileInputDataProvider(const NetworkInputDescriptor* network);

        ~FileInputDataProvider();

        bool load(const std::string& rootFile, const std::string& treeName);

        bool read_event(const int32_t event);

        bool fill_device_buffer(void* buffer);

    private:

        void* mInputBuffer;
        void* mOutputBuffer;
        void* mTransferBuffer;
};

FileInputDataProvider::FileInputDataProvider(const NetworkInputDescriptor* network):InputDataProvider(network),
    mInputBuffer(nullptr), mOutputBuffer(nullptr){}

FileInputDataProvider::~FileInputDataProvider()
{
    if(mInputBuffer != nullptr)
    {
        cudaFreeHost(mInputBuffer);
    }
    if(mOutputBuffer != nullptr)
    {
        cudaFreeHost(mOutputBuffer);
    }
    if(mTransferBuffer != nullptr)
    {
        cudaFreeHost(mTransferBuffer);
    }
}


bool FileInputDataProvider::load(const std::string& rootFile, const std::string& treeName)
{
    auto branchNames = this->mNetworkDescriptor->get_branch_names();
    auto inputSize = branchNames.size() * sizeof(float);
    if(mInputBuffer != nullptr)
    {
        cudaFreeHost(mInputBuffer);
    }
    cudaHostAlloc(&mInputBuffer, inputSize, cudaHostAllocDefault);
    if(mOutputBuffer != nullptr)
    {
        cudaFreeHost(mOutputBuffer);
    }
    cudaHostAlloc(&mInputBuffer, sizeof(float), cudaHostAllocDefault);

    this->open(rootFile, treeName, branchNames, (float*)mInputBuffer, (float*)mOutputBuffer);

    auto transferSize = this->mNetworkDescriptor->host_buffer_size();
    if(mTransferBuffer != nullptr)
    {
        cudaFreeHost(mTransferBuffer);
    }
    cudaHostAlloc(&mTransferBuffer, transferSize, cudaHostAllocDefault);
    return true;
}

bool FileInputDataProvider::read_event(int32_t event)
{
    if(mEventTree == nullptr)
    {
        return false;
    }
    mEventTree->GetEntry(event);
    return true;
}

bool FileInputDataProvider::fill_device_buffer(void* inputBufferDevice)
{
    this->mNetworkDescriptor->fill_host_buffer((float*)mInputBuffer, (float*)mTransferBuffer);
    cudaMemcpy(inputBufferDevice, mTransferBuffer, this->mNetworkDescriptor->host_buffer_size() * sizeof(float), cudaMemcpyHostToDevice);
    return true;
}

/*class GPUInputDataProvider: public InputDataProvider
{
    public:

        GPUInputDataProvider();

        bool load(const std::string& rootFile, const std::string& treeName, const std::vector<std::string>& branchNames, float* buffer);

        bool read_event(const int32_t event);

        bool fill_device_buffer(void* buffer, int32_t size);

    private:

        float* mInputBuffer;
        float* mOutputBuffer;
};

bool GPUInputDataProvider::load(const std::string& rootFile, const std::string& treeName, const std::vector<std::string>& branchNames, float* buffer)
{
    FileInputDataProvider fileInput;
    fileInput.load(rootFile, treeName, branchNames, buffer);
    for(int32_t i = 0; i < nevts; ++i)
    {
        fileInput.read_event(i);

    }
};*/





class GhostDetection
{
    public:
    
        GhostDetection(const std::string& engineFilename, InputDataProvider* dataProvider);

        bool build();

        bool initialize(const std::string& rootFile, const std::string& treeName);

        bool infer(const int32_t nevent, InferenceResult& result);

    private:

        std::string mEngineFilename;

        InputDataProvider* mDataProvider;

        std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> mEngine;

        std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> mContext;
                
        const int32_t mInputSize, mOutputSize;

        void* mInputBufferDevice;
        void* mOutputBufferDevice;
};

GhostDetection::GhostDetection(const std::string& engineFilename, InputDataProvider* dataProvider): mEngineFilename(engineFilename), 
mDataProvider(dataProvider), mEngine(nullptr), mContext(nullptr), mInputSize(17), mOutputSize(1), mInputBufferDevice(nullptr), 
mOutputBufferDevice(nullptr){}

bool GhostDetection::initialize(const std::string& rootFile, const std::string& treeName)
{
    mDataProvider->load(rootFile, treeName);
    // Create TensorRT context
    mContext = std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter>(mEngine->createExecutionContext());
    if (!mContext)
    {
        return false;
    }
}

bool GhostDetection::build()
{
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(nvinfer1::createInferBuilder(ghost_logger));
    if (!builder)
    {
        return false;
    }

    // If necessary check that the GPU supports lower precision
    if ( useFP16 && !builder->platformHasFastFp16() )
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
    if ( useFP16 )
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

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


bool GhostDetection::infer(int32_t nevent, InferenceResult& result)
{
    mDataProvider->read_event(nevent);
    mDataProvider->fill_device_buffer(mInputBufferDevice);
    void* buffers[2];
    buffers[0] = mInputBufferDevice;
    buffers[1] = mOutputBufferDevice;

    // Do inference, batch size 1
    bool status = mContext->executeV2((void* const*)buffers);
    if (!status)
    {
        return false;
    }

//    unsigned truth = ((unsigned*)mOutputBufferHost)[0];
    unsigned truth = 0;
 
    // Memcpy from host input buffers to device input buffers
//    cudaMemcpy(mOutputBufferHost, mOutputBufferDevice, mOutputSize * sizeof(float), cudaMemcpyDeviceToHost);

//    float pred = ((float*)mOutputBufferHost)[0] < 0.5 ? 0 : 1;
    float pred =0;
    if(truth == pred)
    {
        result.true_positives += truth;
        result.true_negatives += (1 - truth);
    }
    else
    {
        result.false_positives += pred;
        result.false_negatives += truth;
    }

    return true;
}


int main(int argc, char* argv[])
{
    NetworkInputDescriptor* networkDescriptor = new GhostNetworkInputDescriptor();
    InputDataProvider* inputDataProvider = new FileInputDataProvider(networkDescriptor);
    GhostDetection ghostinfer("../data/ghost_nn.onnx", inputDataProvider);
    ghostinfer.build();
    ghostinfer.initialize("../data/PrCheckerPlots.root","kalman_validator/kalman_ip_tree");

    InferenceResult result;

    auto start = std::chrono::high_resolution_clock::now();

    for(int32_t i = 0; i < 10000; ++i)
    {
        ghostinfer.infer(i, result);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout<<".............................................."<<std::endl;
    std::cout<<"No. true positives:  "<<result.true_positives<<std::endl;
    std::cout<<"No. true negatives:  "<<result.true_negatives<<std::endl;
    std::cout<<"No. false positives: "<<result.false_positives<<std::endl;
    std::cout<<"No. false negatives: "<<result.false_negatives<<std::endl;
    std::cout<<".............................................."<<std::endl;
    std::cout<<"accuracy:" <<result.accuracy()<<std::endl<<std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout<<"Duration: "<<duration.count()<<" microsec."<<std::endl;
}
